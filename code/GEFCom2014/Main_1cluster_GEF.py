from data_utils import *
from Model_1cluster_GEF import VariationalSeq2Seq_meta

import os
import re
import math
import random
import json
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from sklearn.preprocessing import MinMaxScaler
import zipfile
import holidays

# ============================================================
# Latest GEFCom data processing (fixed)
#   - LOAD is never interpolated
#   - only non-LOAD external numeric features are interpolated
#   - only 7 consecutive valid days are packed into a week
# ============================================================



def _season_from_month(m: int) -> int:
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[m]





# ============================================================
# Future temperature features for GEFCom
#   - no CDD/HDD in this version
#   - thermal expert = current avg temp + future 24h avg-temp path
# ============================================================

def build_oracle_temp_raw_only_1d(temp_1d: np.ndarray, horizon: int):
    """
    For each time t, build the available/forecast future temperature path:
        temp[t], temp[t+1], ..., temp[t+horizon-1]

    In historical experiments, this uses observed future temperature as a
    weather-forecast proxy. Edge padding uses the last available value.
    """
    temp_1d = np.asarray(temp_1d, dtype=float).reshape(-1)
    T = len(temp_1d)
    if T == 0:
        raise ValueError("temp_1d is empty.")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")

    base_idx = np.arange(T)[:, None]
    off_idx = np.arange(horizon)[None, :]
    gather_idx = np.clip(base_idx + off_idx, 0, T - 1)
    future_mat = temp_1d[gather_idx]  # [T, horizon]

    return {f"temp_fc_tplus{h:02d}": future_mat[:, h] for h in range(horizon)}


def build_oracle_temp_raw_only_weekly(temp_weekly: np.ndarray, horizon: int):
    """
    temp_weekly shape: [n_weeks, 168]
    Returns temp_fc_tplus00 ... temp_fc_tplus{horizon-1}, each [n_weeks, 168].
    """
    temp_weekly = np.asarray(temp_weekly, dtype=float)
    if temp_weekly.ndim != 2:
        raise ValueError(
            f"temp_weekly must be 2D [n_weeks, num_in_week], got shape={temp_weekly.shape}"
        )

    n_weeks, num_in_week = temp_weekly.shape
    flat = temp_weekly.reshape(-1)
    feat_flat = build_oracle_temp_raw_only_1d(flat, horizon=horizon)
    return {k: np.asarray(v).reshape(n_weeks, num_in_week) for k, v in feat_flat.items()}


# ============================================================
# Common helpers
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Correct GEFCom2014-L reader from zip
#   - Fix compact timestamp parsing, e.g. "112001 1:00"
#   - Include Task 15 solution to get full Dec. 2011
#   - Pack each year into exactly 52 non-overlap weeks
#   - Return:
#       times, temp_avg, workday, month_feat, load
# ============================================================

def _candidate_gefcom_timestamps(s: str):
    """
    GEFCom compact timestamp examples:
        "112001 1:00"   -> 1/1/2001 1:00
        "1112011 1:00"  -> 1/11/2011 or 11/1/2011, resolved by sequence order

    We return all valid month/day/year candidates.
    """
    s = str(s).strip()
    date_part, time_part = s.split()

    hh, mm = map(int, time_part.split(":"))
    yyyy = int(date_part[-4:])
    md = date_part[:-4]

    candidates = []
    for split in range(1, len(md)):
        try:
            month = int(md[:split])
            day = int(md[split:])
            candidates.append(pd.Timestamp(year=yyyy, month=month, day=day, hour=hh, minute=mm))
        except Exception:
            pass

    if len(candidates) == 0:
        raise ValueError(f"Cannot parse GEFCom timestamp: {s}")

    return sorted(set(candidates))


def _parse_gefcom_timestamp_sequence(timestamp_list, expected_first_date=None):
    """
    Robustly parse one task's timestamp column.

    Important:
    GEFCom load timestamps are hour-ending:
        1:00 means the interval [0:00, 1:00].
    For normal hourly indexing, we subtract 1 hour.
    """
    parsed = []
    prev = None

    if expected_first_date is not None:
        expected_first_date = pd.Timestamp(expected_first_date).date()

    for j, s in enumerate(timestamp_list):
        candidates = _candidate_gefcom_timestamps(s)
        chosen = None

        # First row: use known task start date.
        if j == 0 and expected_first_date is not None:
            matches = [c for c in candidates if c.date() == expected_first_date]
            if len(matches) > 0:
                chosen = matches[0]

        # Later rows: choose the candidate that continues hourly order.
        if chosen is None and prev is not None:
            target = prev + pd.Timedelta(hours=1)
            matches = [c for c in candidates if c == target]

            if len(matches) > 0:
                chosen = matches[0]
            else:
                chosen = min(candidates, key=lambda c: abs((c - target).total_seconds()))

        if chosen is None:
            chosen = candidates[0]

        parsed.append(chosen)
        prev = chosen

    # Convert hour-ending to hour-start.
    return pd.DatetimeIndex(parsed) - pd.Timedelta(hours=1)


def process_seq2seq_data_gefcom_direct_hourly(
    full: pd.DataFrame,
    *,
    output_len=24,
    encoder_len_weeks=1,  # 【新增】灵活控制用前几周的数据，1=168h, 2=336h
    train_years=(2006, 2007, 2008, 2009),
    val_year=2010,
    test_year=2011,
    ext_variables=("temp", "workday", "season", "holiday", "month"),
    device=None,
):
    """
    Direct hourly processing for GEFCom2014.
    For each decoder week:
        encoder = previous `encoder_len_weeks` * 168 hours
        decoder = current 168 hours
    """
    temp_cols = [f"w{i}" for i in range(1, 26)]
    required_cols = ["LOAD"] + temp_cols
    missing = [c for c in required_cols if c not in full.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    full = full.sort_index().copy()

    min_needed_year = min(list(train_years) + [val_year, test_year]) - 1
    max_needed_year = max(list(train_years) + [val_year, test_year])
    start_time = pd.Timestamp(year=min_needed_year, month=1, day=1, hour=0)
    end_time = pd.Timestamp(year=max_needed_year, month=12, day=31, hour=23)
    full = full.loc[start_time:end_time].copy()

    if full[temp_cols].isna().any().any():
        print(full[temp_cols].isna().sum().sort_values(ascending=False).head(10))
        raise ValueError("NaN exists in temperature columns w1...w25.")
    
    # LOAD may be missing before 2005. Individual samples with missing load
    # will be skipped inside add_year(...).

    avg_temp = full[temp_cols].mean(axis=1)
    data = pd.DataFrame(index=full.index)
    data["load"] = full["LOAD"].astype(float)
    data["temp"] = avg_temp.astype(float)
    data["workday"] = (data.index.dayofweek < 5).astype(float)
    data["season"] = np.array(
        [_season_from_month(int(m)) for m in data.index.month],
        dtype=float,
    )

    # Month cyclic labels. These provide finer annual-position information
    # beyond the coarse 4-season regime.
    month_angle = 2.0 * np.pi * (data.index.month.to_numpy() - 1) / 12.0
    data["month_sin"] = np.sin(month_angle)
    data["month_cos"] = np.cos(month_angle)

    # U.S. holiday label, including observed holidays.
    # Example: if July 4 falls on Sunday, Monday may be observed as holiday.
    years_for_holidays = sorted(data.index.year.unique().tolist())
    us_holidays = holidays.country_holidays(
        "US",
        years=years_for_holidays,
        observed=True,
    )
    
    data["holiday"] = np.array(
        [1.0 if ts.date() in us_holidays else 0.0 for ts in data.index],
        dtype=float,
    )

    # Compact future-label summaries for binary calendar variables.
    # We use current label + mean of the next 24 hours instead of a 25-D vector.
    # Example: workday_future24_mean = fraction of hours in t+1,...,t+24 that are workdays.
    def _future_mean_1d(series: pd.Series, horizon: int):
        future_cols = [series.shift(-h) for h in range(1, horizon + 1)]
        return pd.concat(future_cols, axis=1).mean(axis=1)

    data["workday_future24_mean"] = _future_mean_1d(data["workday"], output_len)
    data["holiday_future24_mean"] = _future_mean_1d(data["holiday"], output_len)

    for h in range(output_len):
        data[f"temp_fc_tplus{h:02d}"] = data["temp"].shift(-h)
    data = data.ffill().bfill()
    
    # ------------------------------------------------------------------
    # Flexible expert/external-variable selection.
    # Available variables:
    #   temp    -> [temp, temp_fc_tplus00, ..., temp_fc_tplus23]
    #   workday -> [workday, workday_future24_mean]
    #   season  -> [season]
    #   holiday -> [holiday, holiday_future24_mean]
    #   month   -> [month_sin, month_cos]
    # ------------------------------------------------------------------
    ext_variables = list(ext_variables)
    valid_ext_variables = {"temp", "workday", "season", "holiday", "month"}
    unknown = [v for v in ext_variables if v not in valid_ext_variables]
    if unknown:
        raise ValueError(f"Unknown ext_variables: {unknown}. Valid choices are {sorted(valid_ext_variables)}")

    ext_keys = []
    expert_feature_keys = {}

    if "temp" in ext_variables:
        keys = ["temp"] + [f"temp_fc_tplus{h:02d}" for h in range(output_len)]
        ext_keys.extend(keys)
        expert_feature_keys["temp"] = keys

    if "workday" in ext_variables:
        keys = ["workday", "workday_future24_mean"]
        ext_keys.extend(keys)
        expert_feature_keys["workday"] = keys

    if "season" in ext_variables:
        keys = ["season"]
        ext_keys.extend(keys)
        expert_feature_keys["season"] = keys

    if "holiday" in ext_variables:
        keys = ["holiday", "holiday_future24_mean"]
        ext_keys.extend(keys)
        expert_feature_keys["holiday"] = keys

    if "month" in ext_variables:
        keys = ["month_sin", "month_cos"]
        ext_keys.extend(keys)
        expert_feature_keys["month"] = keys

    feature_cols = ["load"] + ext_keys

    processed = {}
    scalers = {}
    scaling_meta = {}
    for k in feature_cols:
        vec = data[k].to_numpy(dtype=float).reshape(-1, 1)
        sc = MinMaxScaler()
        processed[k] = sc.fit_transform(vec).reshape(-1)
        scalers[k] = sc
        scaling_meta[k] = {
            "scaling": "global_minmax",
            "data_min": float(sc.data_min_[0]),
            "data_max": float(sc.data_max_[0]),
            "feature_range": tuple(float(v) for v in sc.feature_range),
        }

    processed_df = pd.DataFrame(index=data.index)
    for k in feature_cols:
        processed_df[k] = processed[k]

    store = {
        "train": {"X_enc_l": [], "X_enc_ext": [], "X_dec_in_l": [], "X_dec_in_ext": [], "Y_dec_target": []},
        "val": {"X_enc_l": [], "X_enc_ext": [], "X_dec_in_l": [], "X_dec_in_ext": [], "Y_dec_target": []},
        "test": {"X_enc_l": [], "X_enc_ext": [], "X_dec_in_l": [], "X_dec_in_ext": [], "Y_dec_target": []},
    }

    # 【关键修改】计算 encoder 的序列长度
    enc_seq_len = encoder_len_weeks * 168

    def add_year(year, split_name):
        year_start = pd.Timestamp(year=year, month=1, day=1, hour=0)
    
        for k in range(52):
            dec_start = year_start + pd.Timedelta(days=7 * k)
    
            enc_hours = encoder_len_weeks * 168
    
            enc_idx = pd.date_range(
                dec_start - pd.Timedelta(hours=enc_hours),
                periods=enc_hours,
                freq="h",
            )
    
            dec_idx = pd.date_range(
                dec_start,
                periods=168,
                freq="h",
            )
    
            if not enc_idx.isin(processed_df.index).all():
                print(f"[WARN] Skip sample: missing encoder timestamps: {split_name}, year={year}, week={k}")
                continue
    
            if not dec_idx.isin(processed_df.index).all():
                print(f"[WARN] Skip sample: missing decoder timestamps: {split_name}, year={year}, week={k}")
                continue
    
            if dec_idx[0] - enc_idx[-1] != pd.Timedelta(hours=1):
                print(f"[WARN] Skip sample: encoder/decoder are not continuous: {split_name}, year={year}, week={k}")
                continue
    
            enc = processed_df.loc[enc_idx]
            dec = processed_df.loc[dec_idx]
    
            if enc[feature_cols].isna().any().any():
                print(f"[WARN] Skip sample: NaN in encoder: {split_name}, year={year}, week={k}")
                continue
    
            if dec[feature_cols].isna().any().any():
                print(f"[WARN] Skip sample: NaN in decoder: {split_name}, year={year}, week={k}")
                continue
    
            enc_l = enc["load"].to_numpy(dtype=float)
            dec_l = dec["load"].to_numpy(dtype=float)
    
            enc_ext = enc[ext_keys].to_numpy(dtype=float)
            dec_ext = dec[ext_keys].to_numpy(dtype=float)
    
            x_dec_l = dec_l[:168 - output_len]
            x_dec_ext = dec_ext[:168 - output_len]
    
            # Day-ahead protocol: only evaluate forecast origins at daily 00:00 positions.
            # Keep full hourly decoder inputs so that, e.g., the origin t=24 can use
            # the previous 24 hours dec_l[0:24] through the decoder hidden state.
            forecast_indices_np = np.arange(0, 168 - output_len + 1, 24)  # [0,24,...,144]
            y_target = np.stack(
                [dec_l[t:t + output_len] for t in forecast_indices_np],
                axis=0,
            )  # [7, 24]
    
            store[split_name]["X_enc_l"].append(enc_l[:, None])
            store[split_name]["X_enc_ext"].append(enc_ext)
            store[split_name]["X_dec_in_l"].append(x_dec_l[:, None])
            store[split_name]["X_dec_in_ext"].append(x_dec_ext)
            store[split_name]["Y_dec_target"].append(y_target[:, :, None])    
            
    for y in train_years:
        add_year(y, "train")
    add_year(val_year, "val")
    add_year(test_year, "test")

    def _pack(split_name):
        d = store[split_name]
        to_tensor = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).to(device)
        out = {
            "X_enc_l": to_tensor(d["X_enc_l"]),
            "X_enc_ext": to_tensor(d["X_enc_ext"]),
            "X_dec_in_l": to_tensor(d["X_dec_in_l"]),
            "X_dec_in_ext": to_tensor(d["X_dec_in_ext"]),
            "Y_dec_target": to_tensor(d["Y_dec_target"]),
        }
        print(f"[{split_name.upper()}]")
        for k, v in out.items():
            print(f"{k:15s} {tuple(v.shape)}")
        return out

    train_dict = _pack("train")
    val_dict = _pack("val")
    test_dict = _pack("test")

    ext_idx_map = {k: i for i, k in enumerate(ext_keys)}
    expert_specs = []
    for name in ext_variables:
        keys = expert_feature_keys[name]
        expert_specs.append({
            "name": name,
            "indices": [ext_idx_map[k] for k in keys],
            "feature_keys": list(keys),
        })

    return (
        train_dict, val_dict, test_dict, scalers, scaling_meta, ext_keys, expert_specs,
    )


def _read_gefcom2014_hourly_from_zip(
    zip_path: str,
    include_task15_solution: bool = True,
):
    """
    Read all GEFCom2014-L tasks from zip and return hourly dataframe:
        index: hourly datetime
        columns: LOAD, w1, ..., w25
    """
    expected_start = {
        1: "2001-01-01",
        2: "2010-10-01",
        3: "2010-11-01",
        4: "2010-12-01",
    }

    # Task 5 starts from Jan. 2011, Task 6 from Feb. 2011, ..., Task 15 from Nov. 2011.
    for task_id in range(5, 16):
        expected_start[task_id] = f"2011-{task_id - 4:02d}-01"

    frames = []

    with zipfile.ZipFile(zip_path) as z:
        for task_id in range(1, 16):
            name = f"Load/Task {task_id}/L{task_id}-train.csv"
            df = pd.read_csv(z.open(name))

            idx = _parse_gefcom_timestamp_sequence(
                df["TIMESTAMP"].tolist(),
                expected_first_date=expected_start[task_id],
            )

            df = df.drop(columns=["TIMESTAMP"])
            df.insert(0, "date", idx)
            df["__task_id__"] = task_id
            frames.append(df)

        # This is necessary if you want 2011 test = 52 weeks.
        # Task 15 train only gives up to Nov. 2011;
        # Dec. 2011 is in the solution file.
        if include_task15_solution:
            name = "Load/Solution to Task 15/solution15_L_temperature.csv"
            sol = pd.read_csv(z.open(name))

            idx = pd.to_datetime(sol["date"]) + pd.to_timedelta(sol["hour"] - 1, unit="h")

            sol = sol.drop(columns=["date", "hour"])
            if "ZONEID" not in sol.columns:
                sol.insert(0, "ZONEID", 1)

            sol.insert(0, "date", idx)
            sol["__task_id__"] = 16
            frames.append(sol)

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["date", "__task_id__"])

    # Same timestamp appears in multiple tasks.
    # Keep the latest available record.
    full = (
        full.groupby("date", as_index=False)
            .last()
            .set_index("date")
            .sort_index()
    )

    required_cols = ["LOAD"] + [f"w{i}" for i in range(1, 26)]
    missing = [c for c in required_cols if c not in full.columns]
    if missing:
        raise ValueError(f"Missing required GEFCom columns: {missing}")

    full = full[required_cols]

    print("[INFO] Correct GEFCom hourly coverage:")
    print(f"  range: {full.index.min()} to {full.index.max()}")
    for y in range(2005, 2012):
        yy = full[full.index.year == y]
        print(
            f"  {y}: hours={len(yy)}, "
            f"LOAD_nan={int(yy['LOAD'].isna().sum())}"
        )

    return full


def _pack_gefcom_calendar_years_to_52_weeks(
    full: pd.DataFrame,
    start_year: int = 2005,
    end_year: int = 2011,
):
    """
    Pack each year into exactly 52 non-overlap weeks.

    For each year:
        week 0: Jan 1 00:00 to Jan 7 23:00
        week 1: Jan 8 00:00 to Jan 14 23:00
        ...
        week 51: 52nd week

    Remaining 1 day, or 2 days in leap years, is ignored.
    """
    temp_cols = [f"w{i}" for i in range(1, 26)]
    required_cols = ["LOAD"] + temp_cols

    times = []
    temp_avg = []
    workday = []
    month_feat = []
    load = []

    for year in range(start_year, end_year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1, hour=0)

        for k in range(52):
            idx = pd.date_range(
                year_start + pd.Timedelta(days=7 * k),
                periods=168,
                freq="h",
            )

            if not idx.isin(full.index).all():
                raise ValueError(f"Missing timestamps for year={year}, week={k}")

            g = full.loc[idx]

            if g[required_cols].isna().any().any():
                raise ValueError(f"NaN detected for year={year}, week={k}")

            times.append(idx.to_numpy())

            # Average 25 weather stations -> [168]
            temp_avg.append(g[temp_cols].mean(axis=1).to_numpy(dtype=float))

            # workday label: 1 = weekday, 0 = weekend
            workday.append((idx.dayofweek < 5).astype(float))

            # month label: 1,...,12
            month_feat.append(idx.month.astype(float))

            load.append(g["LOAD"].to_numpy(dtype=float))

    times = np.array(times, dtype=object)
    temp_avg = np.array(temp_avg, dtype=float)
    workday = np.array(workday, dtype=float)
    month_feat = np.array(month_feat, dtype=float)
    load = np.array(load, dtype=float)

    print("[INFO] Packed non-overlap weeks:")
    week_years = np.array([pd.DatetimeIndex(t).year[0] for t in times])
    for y in range(start_year, end_year + 1):
        print(f"  {y}: weeks={(week_years == y).sum()}")

    return times, temp_avg, workday, month_feat, load


def get_data_GEFCom2014_avgtemp_latest(
    zip_path: str = "GEFCom2014-L_V2.zip",
    include_task15_solution: bool = True,
):
    """
    Replacement for your original get_data_GEFCom2014_avgtemp_latest().

    Returns:
        times      : [n_weeks], each element has 168 datetimes
        temp_avg   : [n_weeks, 168]
        workday    : [n_weeks, 168]
        month_feat : [n_weeks, 168]
        load       : [n_weeks, 168]
    """
    full = _read_gefcom2014_hourly_from_zip(
        zip_path=zip_path,
        include_task15_solution=include_task15_solution,
    )

    times, temp_avg, workday, month_feat, load = _pack_gefcom_calendar_years_to_52_weeks(
        full=full,
        start_year=2005,
        end_year=2011,
    )

    return times, temp_avg, workday, month_feat, load

def build_warmup_cosine(
    optimizer,
    total_steps: int,
    warmup_ratio: float = 0.04,
    min_lr_ratio: float = 0.1,
):
    """
    Linear warmup + cosine decay scheduler.

    This is stepped once per optimizer update.

    Args:
        optimizer: torch optimizer.
        total_steps: total number of optimizer updates.
        warmup_ratio: fraction of steps used for linear warmup.
        min_lr_ratio: final LR ratio relative to the base LR.

    Behavior:
        - first warmup_ratio * total_steps: LR increases linearly to base LR
        - remaining steps: LR follows cosine decay to base_lr * min_lr_ratio
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def gaussian_icdf(p, device):
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        2 * torch.as_tensor(p, device=device) - 1
    )


def pinball_loss(y, yq, q):
    e = y - yq
    return torch.where(e >= 0, q * e, (q - 1) * e)


def winkler_score(y, L, U, alpha):
    width = (U - L)
    below = (L - y).clamp(min=0.0)
    above = (y - U).clamp(min=0.0)
    return width + (2.0 / alpha) * (below + above)


def crps_gaussian(mu, logvar, target):
    std = (0.5 * logvar).exp()
    z = (target - mu) / std

    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)

    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()


def make_loader(split_dict, batch_size, shuffle):
    ds = TensorDataset(
        split_dict['X_enc_l'],
        split_dict['X_enc_ext'],
        split_dict['X_dec_in_l'],
        split_dict['X_dec_in_ext'],
        split_dict['Y_dec_target'],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _to_year_array(times_weekly: np.ndarray) -> np.ndarray:
    n_weeks = len(times_weekly)
    years = np.zeros((n_weeks, 168), dtype=int)
    for i in range(n_weeks):
        years[i, :] = pd.DatetimeIndex(times_weekly[i]).year.to_numpy()
    return years


# ============================================================
# GEFCom protocol-aligned processing
#   - K = 24
#   - training-set max scaling (unchanged from latest aligned script)
#   - train/val/test = 2006-2009 / 2010 / 2011
# ============================================================

def process_seq2seq_data_gefcom_protocol_minmax(
    feature_dict: Dict[str, np.ndarray],
    times: np.ndarray,
    *,
    output_len=24,
    encoder_len_weeks=1,
    decoder_len_weeks=1,
    num_in_week=168,
    train_years=(2006, 2007, 2008, 2009),
    val_year=2010,
    test_year=2011,
    device=None,
):
    """
    GEFCom protocol split, but preserve the original M2OE2_for_hr.py scaling style:
      - per-feature global MinMaxScaler
      - fit on the full available series for each feature
    This intentionally does NOT use training-set max scaling.
    """
    if "load" not in feature_dict:
        raise ValueError("feature_dict must contain key 'load'")

    n_weeks = feature_dict["load"].shape[0]
    need_weeks = encoder_len_weeks + decoder_len_weeks
    if n_weeks < need_weeks:
        raise ValueError(f"Need ≥{need_weeks} consecutive weeks, found {n_weeks}.")

    processed = {}
    scalers = {}
    scaling_meta = {}
    for k, arr in feature_dict.items():
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            raise ValueError(f"feature '{k}' is empty.")
        vec = arr.reshape(-1)
        sc = MinMaxScaler()
        processed[k] = sc.fit_transform(vec.reshape(-1, 1)).reshape(-1)
        scalers[k] = sc
        scaling_meta[k] = {
            "scaling": "global_minmax",
            "data_min": float(sc.data_min_[0]),
            "data_max": float(sc.data_max_[0]),
            "feature_range": tuple(float(v) for v in sc.feature_range),
        }

    enc_seq_len = encoder_len_weeks * num_in_week
    dec_seq_len = decoder_len_weeks * num_in_week
    L = dec_seq_len - output_len
    if L <= 0:
        raise ValueError("`output_len` must be smaller than decoder sequence length.")

    ext_keys = [k for k in feature_dict.keys() if k != 'load']
    K_ext = len(ext_keys)

    store = {
        "train": {"X_enc_l": [], "X_enc_ext": [], "X_dec_in_l": [], "X_dec_in_ext": [], "Y_dec_target": []},
        "val":   {"X_enc_l": [], "X_enc_ext": [], "X_dec_in_l": [], "X_dec_in_ext": [], "Y_dec_target": []},
        "test":  {"X_enc_l": [], "X_enc_ext": [], "X_dec_in_l": [], "X_dec_in_ext": [], "Y_dec_target": []},
    }

    last_start = n_weeks - need_weeks
    for w in range(last_start + 1):
        enc_start = w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start = enc_end
        dec_end   = dec_start + dec_seq_len

        dec_week_years = pd.DatetimeIndex(times[w + encoder_len_weeks]).year.to_numpy()
        if np.all(np.isin(dec_week_years, list(train_years))):
            split_name = "train"
        elif np.all(dec_week_years == val_year):
            split_name = "val"
        elif np.all(dec_week_years == test_year):
            split_name = "test"
        else:
            continue

        enc_l = processed['load'][enc_start:enc_end]
        dec_full_l = processed['load'][dec_start:dec_end]

        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)
            dec_ext = np.stack([processed[k][dec_start: dec_start + L] for k in ext_keys], axis=-1)
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32)
            dec_ext = np.empty((L, 0), dtype=np.float32)

        targets = np.stack([dec_full_l[i:i + output_len] for i in range(L + 1)], axis=0)

        store[split_name]["X_enc_l"].append(enc_l)
        store[split_name]["X_dec_in_l"].append(dec_full_l[:L])
        store[split_name]["Y_dec_target"].append(targets)
        store[split_name]["X_enc_ext"].append(enc_ext)
        store[split_name]["X_dec_in_ext"].append(dec_ext)

    def _pack(split_name: str):
        d = store[split_name]
        if len(d["X_enc_l"]) == 0:
            raise ValueError(f"No samples created for split '{split_name}'.")
        to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
        out = {
            'X_enc_l': to_tensor(np.array(d["X_enc_l"])).unsqueeze(-1),
            'X_enc_ext': to_tensor(np.array(d["X_enc_ext"])),
            'X_dec_in_l': to_tensor(np.array(d["X_dec_in_l"])).unsqueeze(-1),
            'X_dec_in_ext': to_tensor(np.array(d["X_dec_in_ext"])),
            'Y_dec_target': to_tensor(np.array(d["Y_dec_target"])).unsqueeze(-1),
        }
        print(f"[{split_name.upper()}]")
        for k, v in out.items():
            print(f"{k:15s} {tuple(v.shape)}")
        return out

    train_dict = _pack("train")
    val_dict   = _pack("val")
    test_dict  = _pack("test")
    return train_dict, val_dict, test_dict, scalers, scaling_meta


# ============================================================
# Train / Eval (same structure as aligned script)
# ============================================================

def _enable_deterministic_reparameterize(model, enabled=True):
    """Temporarily make VAE eval deterministic by using z = mu."""
    if (not enabled) or (not hasattr(model, "reparameterize")):
        return None
    old_reparameterize = model.reparameterize
    model.reparameterize = lambda mu, logvar: mu
    return old_reparameterize


def _restore_reparameterize(model, old_reparameterize):
    if old_reparameterize is not None:
        model.reparameterize = old_reparameterize


# ============================================================
# Numerical-stability helpers
#   These guards are intentionally inactive in normal ranges.
#   They only act when log-variance, gradients, or outputs become
#   non-finite / extreme.
# ============================================================

def _all_finite(*tensors) -> bool:
    """Return True only if every provided tensor contains finite values."""
    for t in tensors:
        if t is None:
            continue
        if not torch.is_tensor(t):
            continue
        if not torch.isfinite(t).all():
            return False
    return True


def _model_parameters_finite(model: nn.Module) -> bool:
    """Check whether all trainable parameters are finite."""
    for p in model.parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            return False
    return True


def stable_gaussian_nll_loss(mu, logvar, target):
    """ Gaussian NLL. 模型内部已保证预测 logvar 范围，此处无需再 clamp。 """
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + math.log(2.0 * math.pi) + (target - mu) ** 2 / (var + 1e-12))
    return nll.mean()


def stable_kl_loss(mu, logvar):
    """ Stable VAE KL loss for latent space. """
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))



@torch.no_grad()
def evaluate_rolling_first_step_for_early_stop(
    model,
    val_loader,
    device,
    quantiles=(0.1, 0.5, 0.9),
    alpha=0.1,
    deterministic_eval=True,
    forecast_indices=None,
):
    """
    Validation metric for early stopping.

    This version evaluates only the day-ahead forecast origins produced by the model.

    Expected shapes after squeeze:
        mu      : [B, L, K]
        logvar  : [B, L, K]
        tgt     : [B, L, K]

    Example:
        [B, L, K] = [16, 7, 24] for weekly day-ahead evaluation

    Point-wise metrics are averaged over:
        total_points = B * 7 * 24 in the weekly day-ahead protocol

    Returned metrics:
        val_mse      = mean squared error over all points
        val_nll      = mean Gaussian NLL over all points
        val_crps     = mean Gaussian CRPS over all points
        val_qpin     = mean pinball loss over all points and selected quantiles
        val_winkler  = mean Winkler score over all points
    """

    model.eval()
    old_reparameterize = _enable_deterministic_reparameterize(
        model, deterministic_eval
    )

    mse_sum = 0.0
    nll_sum = 0.0
    crps_sum = 0.0
    qpin_sum = 0.0
    wink_sum = 0.0
    n_points = 0
    printed_horizon_diag = False
    try:
        for batch in val_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = [t.to(device) for t in batch]

            mu_preds, logvar_preds, _, _ = model(
                enc_l, enc_ext, dec_l, dec_ext,
                forecast_indices=forecast_indices,
            )

            mu = mu_preds.squeeze(-1)          # [B, L, K]
            logvar = logvar_preds.squeeze(-1)  # [B, L, K]
            y = tgt.squeeze(-1)                # [B, L, K]

            if mu.shape != y.shape:
                raise ValueError(f"mu shape {mu.shape} does not match target shape {y.shape}")

            if logvar.shape != y.shape:
                raise ValueError(f"logvar shape {logvar.shape} does not match target shape {y.shape}")

            if not _all_finite(mu, logvar, y):
                print("[WARN] Non-finite validation output detected. Returning +inf validation metrics to protect best checkpoint.")
                return {
                    "val_mse": float("inf"),
                    "val_nll": float("inf"),
                    "val_crps": float("inf"),
                    "val_qpin": float("inf"),
                    "val_winkler": float("inf"),
                    "val_n_points": n_points,
                    "val_mode": "day_ahead_nonfinite",
                }


            sigma = torch.exp(0.5 * logvar)

            batch_points = int(mu.numel())
            n_points += batch_points

            # ---------------- MSE ----------------
            mse_sum += ((mu - y) ** 2).sum().item()

            # ---------------- NLL ----------------
            nll = 0.5 * (
                logvar
                + math.log(2.0 * math.pi)
                + (y - mu) ** 2 / (torch.exp(logvar) + 1e-12)
            )
            nll_sum += nll.sum().item()

            # ---------------- CRPS ----------------
            crps = _crps_gaussian_from_mu_sigma(mu, sigma, y)
            crps_sum += crps.sum().item()

            # ---------------- Quantile / Pinball loss ----------------
            for q in quantiles:
                zq = gaussian_icdf(q, device=mu.device)
                yq = mu + sigma * zq
                qpin_sum += pinball_loss(y, yq, q).sum().item()

            # ---------------- Winkler score ----------------
            z = gaussian_icdf(1.0 - alpha / 2.0, device=mu.device)
            lb = mu - z * sigma
            ub = mu + z * sigma
            wink_sum += winkler_score(y, lb, ub, alpha).sum().item()

    finally:
        _restore_reparameterize(model, old_reparameterize)

    if n_points == 0:
        raise ValueError("Validation loader is empty.")

    return {
        "val_mse": mse_sum / n_points,
        "val_nll": nll_sum / n_points,
        "val_crps": crps_sum / n_points,
        "val_qpin": qpin_sum / (n_points * len(quantiles)),
        "val_winkler": wink_sum / n_points,
        "val_n_points": n_points,
        "val_mode": "day_ahead",
    }


def train_model(
    model,
    train_loader,
    epochs,
    lr,
    device,
    top_k=2,
    kl_weight=0.01,
    kl_anneal_epochs=10,
    warmup_epochs=40,
    save_path="best_model.pt",
    val_loader=None,
    patience=1500,
    min_delta=0.0,
    monitor="val_mse",
    deterministic_val=True,
    grad_clip_norm: float = 1.0,
    max_bad_batches_per_epoch: int = 2,
    stop_on_nonfinite_val: bool = True,
    use_adamw: bool = True,
    adam_betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 1e-4,
    use_lr_scheduler: bool = True,
    scheduler_warmup_ratio: float = 0.04,
    scheduler_min_lr_ratio: float = 0.1,
    forecast_indices=None,
):
    """
    Train with validation-based early stopping plus light numerical guards.

    The guards are designed not to affect normal training:
      - logvar is unchanged inside [logvar_min, logvar_max]
      - gradient clipping only activates when gradient norm is larger than grad_clip_norm
      - non-finite batches are skipped instead of updating the model
      - if validation becomes non-finite, training stops gracefully and keeps the best checkpoint
      - AdamW + warmup cosine decay can stabilize late-stage optimization
    """
    if use_adamw:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=adam_betas,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = max(1, epochs * len(train_loader))
    scheduler = None
    if use_lr_scheduler:
        scheduler = build_warmup_cosine(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=scheduler_warmup_ratio,
            min_lr_ratio=scheduler_min_lr_ratio,
        )

    print(
        f"[INFO] Optimizer: {'AdamW' if use_adamw else 'Adam'} | "
        f"lr={lr:.2e} | weight_decay={weight_decay if use_adamw else 0.0:.1e} | "
        f"betas={adam_betas if use_adamw else 'Adam default'}"
    )
    if scheduler is not None:
        print(
            f"[INFO] LR scheduler: warmup-cosine | total_steps={total_steps} | "
            f"warmup_ratio={scheduler_warmup_ratio} | min_lr_ratio={scheduler_min_lr_ratio}"
        )


    best_score = float("inf")
    best_epoch = -1
    no_improve = 0
    stopped_for_numerics = False
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        effective_samples = 0
        bad_batches = 0
        # 【新增】计算当前epoch的KL退火权重 (线性从0增加到kl_weight)
        if kl_anneal_epochs > 0:
            current_kl_weight = kl_weight * min(1.0, ep / kl_anneal_epochs)
        else:
            current_kl_weight = kl_weight  # 如果设为0，则不退火，直接使用kl_weight


        for batch_id, (enc_l, enc_ext, dec_l, dec_ext, tgt) in enumerate(train_loader):
            enc_l, enc_ext, dec_l, dec_ext, tgt = (
                enc_l.to(device), enc_ext.to(device), dec_l.to(device), dec_ext.to(device), tgt.to(device)
            )

            if not _all_finite(enc_l, enc_ext, dec_l, dec_ext, tgt):
                bad_batches += 1
                print(f"[WARN] Epoch {ep} batch {batch_id}: non-finite input/target detected; skipping batch.")
                continue

            optimizer.zero_grad(set_to_none=True)

            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=top_k, warmup_epochs=warmup_epochs,
                forecast_indices=forecast_indices,
            )

            if not _all_finite(mu_preds, logvar_preds, mu_z, logvar_z):
                bad_batches += 1
                print(f"[WARN] Epoch {ep} batch {batch_id}: non-finite model output detected; skipping batch.")
                continue

            nll = stable_gaussian_nll_loss(
                mu_preds,
                logvar_preds,
                tgt,
            )
            kl = stable_kl_loss(
                mu_z,
                logvar_z,
            )
            loss = nll + current_kl_weight * kl

            if not torch.isfinite(loss):
                bad_batches += 1
                print(f"[WARN] Epoch {ep} batch {batch_id}: non-finite loss detected; skipping batch.")
                continue

            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm):
                    bad_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    print(f"[WARN] Epoch {ep} batch {batch_id}: non-finite gradient norm detected; skipping optimizer step.")
                    continue

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if not _model_parameters_finite(model):
                print(f"[ERROR] Epoch {ep} batch {batch_id}: non-finite model parameter detected after optimizer step.")
                print("[ERROR] Stopping training to protect the saved best checkpoint.")
                stopped_for_numerics = True
                break

            running += loss.item() * enc_l.size(0)
            effective_samples += enc_l.size(0)

        if stopped_for_numerics:
            break

        if bad_batches > max_bad_batches_per_epoch:
            print(
                f"[ERROR] Epoch {ep}: skipped {bad_batches} bad batches, "
                f"which exceeds max_bad_batches_per_epoch={max_bad_batches_per_epoch}."
            )
            print("[ERROR] Stopping training to protect the saved best checkpoint.")
            stopped_for_numerics = True
            break

        if effective_samples == 0:
            print(f"[ERROR] Epoch {ep}: all batches were skipped. Stopping training.")
            stopped_for_numerics = True
            break

        train_loss = running / effective_samples

        if val_loader is not None:
            val_metrics = evaluate_rolling_first_step_for_early_stop(
                model,
                val_loader,
                device,
                deterministic_eval=deterministic_val,
                forecast_indices=forecast_indices,
            )
            score = float(val_metrics[monitor])
        else:
            val_metrics = {}
            score = train_loss
            monitor = "train_loss"

        if not math.isfinite(score):
            print(f"[WARN] Epoch {ep}: non-finite validation score for {monitor}.")
            if stop_on_nonfinite_val:
                print("[WARN] Stopping training and keeping the saved best checkpoint.")
                stopped_for_numerics = True
                break

        improved = score < (best_score - min_delta)
        if improved:
            best_score = score
            best_epoch = ep
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {ep} | {monitor} {best_score:.6f} | train loss {train_loss:.6f}")
        else:
            no_improve += 1

        if ep == 1 or ep % 5 == 0 or ep == epochs or improved or bad_batches > 0:
            current_lr = optimizer.param_groups[0]["lr"]
            val_msg = ""
            if val_metrics:
                val_msg = (
                    f" | val_mse={val_metrics['val_mse']:.6f}"
                    f" val_crps={val_metrics['val_crps']:.6f}"
                    f" val_qpin={val_metrics['val_qpin']:.6f}"
                    f" val_winkler={val_metrics['val_winkler']:.6f}"
                )
            bad_msg = f" | bad_batches={bad_batches}" if bad_batches > 0 else ""
            print(
                f"Epoch {ep:4d}/{epochs} | train loss: {train_loss:.6f}"
                f" | lr={current_lr:.2e}"
                f" | kl_w={current_kl_weight:.2e}" # 新增这一行
                f" | current {monitor}: {score:.6f}"
                f" | best {monitor}: {best_score:.6f} (ep {best_epoch})"
                f" | no_improve={no_improve}/{patience}{bad_msg}{val_msg}"
            )

        if patience is not None and patience > 0 and no_improve >= patience:
            print(
                f"\n⏹ Early stopping at epoch {ep}: no validation improvement for {patience} epochs. "
                f"Best epoch {best_epoch} | best {monitor} {best_score:.6f}"
            )
            break

    if stopped_for_numerics and os.path.isfile(save_path):
        print(f"[INFO] Reloading best checkpoint after numerical stop: {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

    print(f"\n🏁 Done. Best epoch {best_epoch} | best {monitor} {best_score:.6f}")
    return model
def _crps_gaussian_from_mu_sigma(mu, sigma, target):
    sigma = sigma.clamp(min=1e-8)
    z = (target - mu) / sigma
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def _get_load_inverse_params_from_scaler(scalers, scaling_meta):
    """Return scale and shift for raw = normalized * scale + shift."""
    if scalers is not None and isinstance(scalers, dict) and "load" in scalers:
        sc = scalers["load"]
        data_min = float(sc.data_min_[0])
        data_max = float(sc.data_max_[0])
        fr_min, fr_max = sc.feature_range
        scale = (data_max - data_min) / (fr_max - fr_min)
        shift = data_min - fr_min * scale
        return scale, shift

    if scaling_meta is not None and "load" in scaling_meta:
        meta = scaling_meta["load"]
        data_min = float(meta["data_min"])
        data_max = float(meta["data_max"])
        fr_min, fr_max = meta.get("feature_range", (0.0, 1.0))
        scale = (data_max - data_min) / (fr_max - fr_min)
        shift = data_min - fr_min * scale
        return scale, shift

    raise ValueError("Need scalers['load'] or scaling_meta['load'] to inverse-transform load.")


@torch.no_grad()
def _crps_gaussian_from_mu_sigma(mu, sigma, target):
    """
    Element-wise CRPS for Gaussian predictive distribution.
    Returns tensor with same shape as mu/target.
    """
    sigma = sigma.clamp(min=1e-8)
    z = (target - mu) / sigma

    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)

    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps

@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    device,
    model_path="model.pt",
    quantiles=(0.1, 0.5, 0.9),
    alpha=0.1,
    load_scale=None,
    load_shift=None,
    compute_raw=True,
    compute_paper3=True,
    compute_paper_daily=True,  # 新增：是否计算对齐论文的日前评估
    deterministic_eval=True,
    forecast_indices=None,
):
    """
    Day-ahead evaluation only.

    Model output shape after squeeze:
        mu_preds      : [B, 7, 24]
        logvar_preds  : [B, 7, 24]
        tgt           : [B, 7, 24]

    where the 7 origins correspond to t = 0, 24, ..., 144 in each week.
    """

    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    old_reparameterize = _enable_deterministic_reparameterize(model, deterministic_eval)

    # ---------------- normalized metric sums ----------------
    norm_mse_sum = 0.0
    norm_nll_sum = 0.0
    norm_crps_sum = 0.0
    norm_qpin_sum = 0.0
    norm_wink_sum = 0.0
    norm_n_points = 0
    printed_horizon_diag = False
    norm_pve_abs_sum = 0.0
    norm_pve_pct_sum = 0.0
    norm_n_samples = 0

    # ---------------- raw-scale metric sums ----------------
    raw_enabled = compute_raw and (load_scale is not None) and (load_shift is not None)

    raw_mse_sum = 0.0
    raw_nll_sum = 0.0
    raw_crps_sum = 0.0
    raw_qpin_sum = 0.0
    raw_wink_sum = 0.0
    raw_n_points = 0

    raw_pve_abs_sum = 0.0
    raw_pve_pct_sum = 0.0
    raw_n_samples = 0

    # ---------------- paper-style raw metrics (All horizons) ----------------
    paper_pinball_sum = 0.0
    paper_winkler50_sum = 0.0
    paper_winkler90_sum = 0.0
    paper_n_points = 0
    all_qs = [q / 100.0 for q in range(1, 100)]

    # ---------------- paper-style DAILY raw metrics (新增) ----------------
    paper_daily_pinball_sum = 0.0
    paper_daily_winkler50_sum = 0.0
    paper_daily_winkler90_sum = 0.0
    paper_daily_n_points = 0

    try:
        for batch in test_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = [t.to(device) for t in batch]
            B = enc_l.size(0)

            mu_preds, logvar_preds, _, _ = model(
                enc_l, enc_ext, dec_l, dec_ext,
                forecast_indices=forecast_indices,
            )

            mu = mu_preds.squeeze(-1)          # [B, L, K]
            logvar = logvar_preds.squeeze(-1)  # [B, L, K]
            y = tgt.squeeze(-1)                # [B, L, K]

            if mu.shape != y.shape:
                raise ValueError(f"mu shape {mu.shape} does not match target shape {y.shape}")

            if logvar.shape != y.shape:
                raise ValueError(f"logvar shape {logvar.shape} does not match target shape {y.shape}")

            if not _all_finite(mu, logvar, y):
                print("[WARN] Non-finite evaluation output detected; skipping this batch.")
                continue


            sigma = torch.exp(0.5 * logvar)

            # Diagnostic: print once per evaluate_model call.
            if not printed_horizon_diag:
                sqerr_h = ((mu - y) ** 2).mean(dim=(0, 1))   # [K]
                rmse_h = torch.sqrt(sqerr_h)
                crps_h = _crps_gaussian_from_mu_sigma(mu, sigma, y).mean(dim=(0, 1))  # [K]

                print("\n[DIAG] Per-horizon normalized RMSE:")
                for h in range(mu.size(2)):
                    print(f"h={h+1:02d}: RMSE={rmse_h[h].item():.6f}, CRPS={crps_h[h].item():.6f}")
                printed_horizon_diag = True

            n_points_batch = int(mu.numel())

            # ==================================================
            # Normalized day-ahead metrics
            # ==================================================
            norm_n_points += n_points_batch

            err2 = (mu - y) ** 2
            norm_mse_sum += err2.sum().item()

            nll = 0.5 * (
                logvar
                + math.log(2.0 * math.pi)
                + (y - mu) ** 2 / (torch.exp(logvar) + 1e-12)
            )
            norm_nll_sum += nll.sum().item()

            crps = _crps_gaussian_from_mu_sigma(mu, sigma, y)
            norm_crps_sum += crps.sum().item()

            for q in quantiles:
                zq = gaussian_icdf(q, device=mu.device)
                yq = mu + sigma * zq
                norm_qpin_sum += pinball_loss(y, yq, q).sum().item()

            z = gaussian_icdf(1.0 - alpha / 2.0, device=mu.device)
            lb = mu - z * sigma
            ub = mu + z * sigma
            norm_wink_sum += winkler_score(y, lb, ub, alpha).sum().item()

            # Peak over all rolling origins and all horizons per sample
            y_flat = y.reshape(B, -1)
            mu_flat = mu.reshape(B, -1)

            p_true = y_flat.max(dim=1).values
            p_pred = mu_flat.max(dim=1).values
            p_diff = (p_pred - p_true).abs()

            norm_pve_abs_sum += p_diff.sum().item()
            norm_pve_pct_sum += (p_diff / (p_true.abs() + 1e-12)).sum().item()
            norm_n_samples += B

            # ==================================================
            # Raw-scale day-ahead metrics
            # ==================================================
            if raw_enabled:
                mu_raw = mu * load_scale + load_shift
                y_raw = y * load_scale + load_shift
                sigma_raw = sigma * abs(load_scale)
                logvar_raw = 2.0 * torch.log(sigma_raw.clamp(min=1e-8))

                raw_n_points += int(mu_raw.numel())

                raw_mse_sum += ((mu_raw - y_raw) ** 2).sum().item()

                raw_nll = 0.5 * (
                    logvar_raw
                    + math.log(2.0 * math.pi)
                    + (y_raw - mu_raw) ** 2 / (torch.exp(logvar_raw) + 1e-12)
                )
                raw_nll_sum += raw_nll.sum().item()

                raw_crps = _crps_gaussian_from_mu_sigma(mu_raw, sigma_raw, y_raw)
                raw_crps_sum += raw_crps.sum().item()

                for q in quantiles:
                    zq = gaussian_icdf(q, device=mu_raw.device)
                    yq_raw = mu_raw + sigma_raw * zq
                    raw_qpin_sum += pinball_loss(y_raw, yq_raw, q).sum().item()

                z_raw = gaussian_icdf(1.0 - alpha / 2.0, device=mu_raw.device)
                lb_raw = mu_raw - z_raw * sigma_raw
                ub_raw = mu_raw + z_raw * sigma_raw
                raw_wink_sum += winkler_score(y_raw, lb_raw, ub_raw, alpha).sum().item()

                y_raw_flat = y_raw.reshape(B, -1)
                mu_raw_flat = mu_raw.reshape(B, -1)

                p_true_raw = y_raw_flat.max(dim=1).values
                p_pred_raw = mu_raw_flat.max(dim=1).values
                raw_diff = (p_pred_raw - p_true_raw).abs()

                raw_pve_abs_sum += raw_diff.sum().item()
                raw_pve_pct_sum += (raw_diff / (p_true_raw.abs() + 1e-12)).sum().item()
                raw_n_samples += B

                if compute_paper3:
                    for q in all_qs:
                        zq = gaussian_icdf(q, device=mu_raw.device)
                        yq_raw = mu_raw + sigma_raw * zq
                        paper_pinball_sum += pinball_loss(y_raw, yq_raw, q).sum().item()

                    z50 = gaussian_icdf(0.75, device=mu_raw.device)
                    l50 = mu_raw - z50 * sigma_raw
                    u50 = mu_raw + z50 * sigma_raw
                    paper_winkler50_sum += winkler_score(y_raw, l50, u50, alpha=0.50).sum().item()

                    z90 = gaussian_icdf(0.95, device=mu_raw.device)
                    l90 = mu_raw - z90 * sigma_raw
                    u90 = mu_raw + z90 * sigma_raw
                    paper_winkler90_sum += winkler_score(y_raw, l90, u90, alpha=0.10).sum().item()

                    paper_n_points += int(mu_raw.numel())

                # ==================================================
                # Paper-style DAILY raw metrics (新增日前评估逻辑)
                # ==================================================
                if compute_paper_daily:
                    L_dim = mu_raw.size(1)  # already 7 in day-ahead mode
                    K_dim = mu_raw.size(2)  # 24
                    
                    # Model output is already restricted to day-ahead origins.
                    day_start_indices = list(range(L_dim))
                    
                    # Flatten: [B, 7, 24] -> [B*7, 24]
                    mu_d = mu_raw[:, day_start_indices, :].reshape(-1, K_dim)
                    sigma_d = sigma_raw[:, day_start_indices, :].reshape(-1, K_dim)
                    y_d = y_raw[:, day_start_indices, :].reshape(-1, K_dim)
                    
                    n_points_daily = mu_d.numel()
                    paper_daily_n_points += n_points_daily
                    
                    # 计算 99 个分位数的 Pinball Loss
                    for q in all_qs:
                        zq = gaussian_icdf(q, device=mu_d.device)
                        yq_d = mu_d + sigma_d * zq
                        paper_daily_pinball_sum += pinball_loss(y_d, yq_d, q).sum().item()

                    # 计算 Winkler 50%
                    z50 = gaussian_icdf(0.75, device=mu_d.device)
                    l50 = mu_d - z50 * sigma_d
                    u50 = mu_d + z50 * sigma_d
                    paper_daily_winkler50_sum += winkler_score(y_d, l50, u50, alpha=0.50).sum().item()

                    # 计算 Winkler 90%
                    z90 = gaussian_icdf(0.95, device=mu_d.device)
                    l90 = mu_d - z90 * sigma_d
                    u90 = mu_d + z90 * sigma_d
                    paper_daily_winkler90_sum += winkler_score(y_d, l90, u90, alpha=0.10).sum().item()

    finally:
        _restore_reparameterize(model, old_reparameterize)

    # ======================================================
    # Final normalized averages
    # ======================================================
    if norm_n_points == 0:
        raise ValueError("No finite batches were available for evaluation.")

    test_mse = norm_mse_sum / norm_n_points
    test_rmse = math.sqrt(test_mse)
    test_nll = norm_nll_sum / norm_n_points
    test_crps = norm_crps_sum / norm_n_points
    test_qpin = norm_qpin_sum / (norm_n_points * len(quantiles))
    test_wink = norm_wink_sum / norm_n_points

    test_pve_abs = norm_pve_abs_sum / norm_n_samples
    test_pve_pct = norm_pve_pct_sum / norm_n_samples

    print("[INFO] Evaluation mode: day-ahead origins only")
    print(f"[INFO] Deterministic VAE eval: {deterministic_eval}")
    print(f"[INFO] Normalized n_points: {norm_n_points}")
    print(f"🧪 Normalized Test MSE         : {test_mse:.6f}")
    print(f"🧪 Normalized Test RMSE        : {test_rmse:.6f}")
    print(f"🧪 Normalized Test NLL         : {test_nll:.6f}")
    print(f"🧪 Normalized Test CRPS        : {test_crps:.6f}")
    print(f"🧪 Normalized Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
    print(f"🧪 Normalized Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
    print(f"🧪 Normalized Peak Value Error : {test_pve_abs:.6f} (absolute)")
    print(f"🧪 Normalized Peak Value Error%: {100.0 * test_pve_pct:.2f}% of true peak")

    normalized_results = {
        "MSE": test_mse,
        "RMSE": test_rmse,
        "NLL": test_nll,
        "CRPS": test_crps,
        "QuantileLoss": test_qpin,
        "WinklerScore": test_wink,
        "PeakValueError_abs": test_pve_abs,
        "PeakValueError_pct": 100.0 * test_pve_pct,
        "n_points": norm_n_points,
        "n_samples": norm_n_samples,
        "mode": "day_ahead",
    }

    # ======================================================
    # Final raw-scale averages
    # ======================================================
    raw_results = None
    paper3_results = None
    paper_daily_results = None  # 新增

    if raw_enabled:
        raw_mse = raw_mse_sum / raw_n_points
        raw_rmse = math.sqrt(raw_mse)
        raw_nll = raw_nll_sum / raw_n_points
        raw_crps = raw_crps_sum / raw_n_points
        raw_qpin = raw_qpin_sum / (raw_n_points * len(quantiles))
        raw_wink = raw_wink_sum / raw_n_points

        raw_pve_abs = raw_pve_abs_sum / raw_n_samples
        raw_pve_pct = raw_pve_pct_sum / raw_n_samples

        raw_results = {
            "raw_MSE": raw_mse,
            "raw_RMSE": raw_rmse,
            "raw_NLL": raw_nll,
            "raw_CRPS": raw_crps,
            "raw_QuantileLoss": raw_qpin,
            "raw_WinklerScore_90PI": raw_wink,
            "raw_PeakValueError_abs": raw_pve_abs,
            "raw_PeakValueError_pct": 100.0 * raw_pve_pct,
            "raw_n_points": raw_n_points,
            "raw_n_samples": raw_n_samples,
            "mode": "day_ahead",
        }

        print(f"\n[INFO] load inverse: raw = normalized * {load_scale:.6f} + {load_shift:.6f}")
        print(f"[INFO] Raw n_points: {raw_n_points}")
        print(f"🧪 Raw Test MSE              : {raw_mse:.6f}")
        print(f"🧪 Raw Test RMSE             : {raw_rmse:.6f}")
        print(f"🧪 Raw Test NLL              : {raw_nll:.6f}")
        print(f"🧪 Raw Test CRPS             : {raw_crps:.6f}")
        print(f"🧪 Raw Test QuantileLoss     : {raw_qpin:.6f}  (avg over q={list(quantiles)})")
        print(f"🧪 Raw Test WinklerScore     : {raw_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
        print(f"🧪 Raw Peak Value Error      : {raw_pve_abs:.6f} (absolute)")
        print(f"🧪 Raw Peak Value Error%     : {100.0 * raw_pve_pct:.2f}% of true peak")

        if compute_paper3 and paper_n_points > 0:
            pinball_allq = paper_pinball_sum / (paper_n_points * len(all_qs))
            winkler_50 = paper_winkler50_sum / paper_n_points
            winkler_90 = paper_winkler90_sum / paper_n_points

            paper3_results = {
                "pinball_allq": pinball_allq,
                "winkler_50": winkler_50,
                "winkler_90": winkler_90,
                "n_points": paper_n_points,
                "mode": "day_ahead",
                "scale": "raw",
            }

            print("\n=== Paper-style metrics | All horizons | Raw scale ===")
            print(f"pinball_allq: {pinball_allq:.6f}")
            print(f"winkler_50  : {winkler_50:.6f}")
            print(f"winkler_90  : {winkler_90:.6f}")
            print(f"n_points    : {paper_n_points}")

        # ======================================================
        # Final paper-style DAILY averages (新增打印)
        # ======================================================
        if compute_paper_daily and paper_daily_n_points > 0:
            pinball_allq_daily = paper_daily_pinball_sum / (paper_daily_n_points * len(all_qs))
            winkler_50_daily = paper_daily_winkler50_sum / paper_daily_n_points
            winkler_90_daily = paper_daily_winkler90_sum / paper_daily_n_points

            paper_daily_results = {
                "pinball_allq_daily": pinball_allq_daily,
                "winkler_50_daily": winkler_50_daily,
                "winkler_90_daily": winkler_90_daily,
                "n_points_daily": paper_daily_n_points,
                "mode": "day_ahead",
                "scale": "raw",
            }

            print("\n=== Paper-style metrics | Day-ahead (00:00 only) | Raw scale ===")
            print(f"pinball_allq_daily: {pinball_allq_daily:.6f}")
            print(f"winkler_50_daily  : {winkler_50_daily:.6f}")
            print(f"winkler_90_daily  : {winkler_90_daily:.6f}")
            print(f"n_points_daily    : {paper_daily_n_points}")

    return {
        "normalized": normalized_results,
        "raw": raw_results,
        "paper3_raw_all_horizons": paper3_results,
        "paper3_raw_day_ahead": paper_daily_results,  # 新增返回
    }

# ============================================================
# Main: future-temperature run with AdamW + warmup-cosine scheduler + gradient clipping
# ============================================================
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    batch_size = 13
    epochs = 1500
    patience = 120
    lr = 1e-3
    optimizer_name = "AdamW"
    adam_betas = (0.9, 0.95)
    weight_decay = 1e-4
    grad_clip_norm = 1.0
    use_lr_scheduler = True
    scheduler_warmup_ratio = 0.04
    scheduler_min_lr_ratio = 0.1
    
    kl_weight = 1e-4
    kl_anneal_epochs = 50  # 【新增】设定KL退火周期，比如前50个epoch线性增加KL权重
    
    print('logvar_max = -3.8')
    
    encoder_len_weeks = 1 

    xprime_dim = 16
    hidden_dim = 64
    latent_dim = 64
    num_layers = 1
    output_len = 24   # K = 24
    TEMP_FC_HORIZON = output_len

    # Choose which external variables become meta experts.
    # Available: "temp", "workday", "season", "holiday", "month".
    # workday/holiday use compact features: current label + next-24h mean label.
    ext_variables = ["temp", "workday", "holiday", "month"]
    # None means use all selected experts. You can also set an integer, e.g., 3.
    top_k = 4
    warmup_ep = 40
    
    # 【新增】定义预测 logvar 的硬边界超参数
    logvar_min = -10.0   # 对应 -9 std ≈ 0.011
    logvar_max = -3.8 # 对应 -3 std ≈ 0.223， -6 对应 std 0.05， -4.6 对应 std 0.1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[INFO] Batch size = {batch_size}")

    data_name = "GEFCom2014"
    ext_tag = "ext" + "-".join(ext_variables)
    opt_tag = (
        f"{optimizer_name}_warmcos{int(scheduler_warmup_ratio * 100):02d}"
        f"_minlr{str(scheduler_min_lr_ratio).replace('.', 'p')}"
        f"_clip{str(grad_clip_norm).replace('.', 'p')}"
        f"_wd{weight_decay:.0e}"
    )
    model_name = "M2OE2_1cluster"
    model_path = "M2OE2_1cluster.pt"
    results_path = "M2OE2_1cluster_eval_results.json"
    print(f"[INFO] model_name: {model_name}")
    print(f"[INFO] model_path: {model_path}")
    print(f"[INFO] results_path: {results_path}")

    # Latest cleaned data + 25-station average temperature.
    # Future temperature path is added as exogenous forecast input.
    # No CDD/HDD is used in this GEFCom version.

    # 【新增】灵活控制 Encoder 使用前几周的数据 (1 或 2)

    zip_path = "GEFCom2014-L_V2.zip"
    full = _read_gefcom2014_hourly_from_zip(
        zip_path=zip_path,
        include_task15_solution=True,
    )
    (
        train_data,
        val_data,
        test_data,
        scalers,
        scaling_meta,
        ext_keys,
        expert_specs,
    ) = process_seq2seq_data_gefcom_direct_hourly(
        full=full,
        output_len=output_len,
        encoder_len_weeks=encoder_len_weeks,  # 【新增】传入参数
        train_years=(2006, 2007, 2008, 2009),
        val_year=2010,
        test_year=2011,
        ext_variables=ext_variables,
        device=device,
    )


    print(f"[INFO] ext_variables: {ext_variables}")
    print(f"[INFO] Raw external feature keys ({len(ext_keys)}): {ext_keys}")
    print("[INFO] expert_specs:")
    for spec in expert_specs:
        print(f"  - {spec['name']}: dim={len(spec['indices'])}, indices={spec['indices']}, keys={spec.get('feature_keys', [])}")

    # If top_k is None, use all selected experts.
    # If top_k is larger than the number of selected experts, warn and clip.
    num_selected_experts = len(expert_specs)
    requested_top_k = top_k

    if requested_top_k is None:
        top_k = num_selected_experts
        print(f"[INFO] top_k=None -> use all selected experts: top_k={top_k}")
    else:
        requested_top_k = int(requested_top_k)
        if requested_top_k > num_selected_experts:
            print(
                f"[WARN] Requested top_k={requested_top_k} is larger than "
                f"num_selected_experts={num_selected_experts}. "
                f"Clipping top_k to {num_selected_experts}."
            )
            top_k = num_selected_experts
        elif requested_top_k <= 0:
            print(
                f"[WARN] Requested top_k={requested_top_k} is <= 0. "
                f"Using all selected experts instead: top_k={num_selected_experts}."
            )
            top_k = num_selected_experts
        else:
            top_k = requested_top_k

        print(f"[INFO] top_k used by gating: {top_k} / {num_selected_experts}")

    print("[INFO] global MinMax scaling meta:")
    for k, v in scaling_meta.items():
        print(f"  {k}: {v}")

    input_dim = 1
    output_dim = 1

    n_externals = train_data['X_enc_ext'].shape[-1]
    print(f"K_ext (number of external features) = {n_externals}")
    print(f"[INFO] Raw external feature keys used = {ext_keys}")
    print(f"[INFO] Number of meta experts = {len(expert_specs)}")
    print("[INFO] Scaling style actually used: per-feature global MinMaxScaler (same as M2OE2_for_hr.py)")
    if n_externals != len(ext_keys):
        raise ValueError(f"n_externals={n_externals} but len(ext_keys)={len(ext_keys)}")

    load_scale, load_shift = _get_load_inverse_params_from_scaler(scalers, scaling_meta)
    print(f"[INFO] load inverse transform: raw = normalized * {load_scale:.6f} + {load_shift:.6f}")

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    val_loader = make_loader(val_data, batch_size, shuffle=False)
    test_loader = make_loader(test_data, batch_size, shuffle=False)

    # Day-ahead forecast origins within each 168-hour decoder week.
    # The decoder still consumes all 144 hourly inputs, but only produces/evaluates
    # forecasts at t = 0, 24, 48, 72, 96, 120, 144.
    forecast_indices = torch.arange(0, 168 - output_len + 1, 24, device=device)
    print(f"[INFO] Day-ahead forecast_indices: {forecast_indices.detach().cpu().tolist()}")

    model = VariationalSeq2Seq_meta(
        xprime_dim=xprime_dim,
        input_dim=input_dim,
        hidden_size=hidden_dim,
        latent_size=latent_dim,
        output_len=output_len,
        n_externals=n_externals,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1,
        expert_specs=expert_specs,
        logvar_min=logvar_min,   # 【新增】传入模型
        logvar_max=logvar_max,   # 【新增】传入模型
    ).to(device)

    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device,
                    top_k=top_k, kl_weight=kl_weight, kl_anneal_epochs=kl_anneal_epochs, warmup_epochs=warmup_ep,
                    save_path=model_path, val_loader=val_loader, patience=patience,
                    min_delta=0.0, monitor="val_mse", deterministic_val=True,
                    grad_clip_norm=grad_clip_norm,
                    use_adamw=(optimizer_name.lower() == "adamw"),
                    adam_betas=adam_betas,
                    weight_decay=weight_decay,
                    use_lr_scheduler=use_lr_scheduler,
                    scheduler_warmup_ratio=scheduler_warmup_ratio,
                    scheduler_min_lr_ratio=scheduler_min_lr_ratio,
                    forecast_indices=forecast_indices)
    else:
        print(f"[✓] Found '{model_path}', loading weights.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    import time
    time1 = time.time()
    print("\n=== Validation ===")
    val_results = evaluate_model(model, val_loader, device, model_path=model_path,
                   load_scale=load_scale, load_shift=load_shift, compute_raw=True, compute_paper3=True,
                   deterministic_eval=True, forecast_indices=forecast_indices)
    print("\n=== Test ===")
    test_results = evaluate_model(model, test_loader, device, model_path=model_path, 
                   load_scale=load_scale, load_shift=load_shift, compute_raw=True, compute_paper3=True,
                   deterministic_eval=True, forecast_indices=forecast_indices)
    time2 = time.time()
    elapsed_sec = time2 - time1
    print("time", elapsed_sec)

    eval_summary = {
        "model_name": model_name,
        "model_path": model_path,
        "optimizer": optimizer_name,
        "lr": lr,
        "adam_betas": list(adam_betas),
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "use_lr_scheduler": use_lr_scheduler,
        "scheduler": {
            "type": "warmup_cosine",
            "warmup_ratio": scheduler_warmup_ratio,
            "min_lr_ratio": scheduler_min_lr_ratio,
        },
        "batch_size": batch_size,
        "epochs": epochs,
        "kl_weight": kl_weight,
        "external_raw_dim": int(n_externals),
        "ext_variables": list(ext_variables),
        "ext_keys": list(ext_keys),
        "num_meta_experts": len(expert_specs),
        "expert_specs": expert_specs,
        "top_k": int(top_k),
        "elapsed_seconds": elapsed_sec,
        "validation": val_results,
        "test": test_results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[INFO] Saved evaluation results to: {results_path}")
