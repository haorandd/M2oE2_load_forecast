from data_utils import *
from model_v1 import *

import os
import re
import math
import random
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Latest GEFCom data processing (fixed)
#   - LOAD is never interpolated
#   - only non-LOAD external numeric features are interpolated
#   - only 7 consecutive valid days are packed into a week
# ============================================================

def _parse_compact_timestamp(s: str):
    if not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    t = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.notna(t):
        return t
    m = re.match(r"^\s*(\d{1,2})(\d{1,2})(\d{4})\s+(\d{1,2}):(\d{2})\s*$", s)
    if m:
        mm, dd, yyyy, HH, MM = m.groups()
        try:
            return pd.Timestamp(year=int(yyyy), month=int(mm), day=int(dd), hour=int(HH), minute=int(MM))
        except Exception:
            return pd.NaT
    return pd.NaT


def _season_from_month(m: int) -> int:
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[m]


def _read_gefcom_task_csv(path_csv: str, task_id: int, timestamp_col: str = "TIMESTAMP") -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    parsed = df[timestamp_col].apply(_parse_compact_timestamp)
    mask = ~parsed.isna()
    df = df.loc[mask].copy()
    parsed = parsed[mask]
    df.insert(0, "date", parsed.values)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop(columns=[timestamp_col])

    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    ordered = []
    if "ZONEID" in num_cols_all:
        ordered.append("ZONEID")
    if "LOAD" in num_cols_all:
        ordered.append("LOAD")
    wnames = sorted([c for c in num_cols_all if re.fullmatch(r"w\d+", c)], key=lambda s: int(s[1:]))
    ordered.extend(wnames)
    remaining = [c for c in num_cols_all if c not in set(ordered)]
    ordered.extend(remaining)
    num_cols = ordered

    out = df[["date"] + num_cols].copy()
    out["__task_id__"] = task_id
    return out


def _pack_hourly_df_to_weeks(df_hourly: pd.DataFrame, feature_names: List[str]):
    """
    Important:
      - LOAD must NEVER be interpolated
      - only 7 consecutive valid days are packed into a week
    """
    df = df_hourly.sort_index().copy()

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)

    load_col = "LOAD" if "LOAD" in feature_names else None
    interp_cols = [c for c in feature_names if c != load_col]

    if interp_cols:
        df[interp_cols] = df[interp_cols].interpolate("time").ffill().bfill()

    df["workday"] = np.where(df.index.dayofweek < 5, 0, 1).astype(int)  # 0=weekday, 1=weekend
    df["season"] = np.array([_season_from_month(m) for m in df.index.month], dtype=int)

    valid_days = []
    skipped_days = 0
    for day, g in df.groupby(df.index.date):
        start = pd.Timestamp(day)
        hidx = pd.date_range(start, periods=24, freq="h")
        g = g.reindex(hidx)

        if interp_cols:
            g[interp_cols] = g[interp_cols].interpolate("time").ffill().bfill()

        if len(g) != 24:
            skipped_days += 1
            continue
        if load_col is not None and g[load_col].isna().any():
            skipped_days += 1
            continue
        if interp_cols and g[interp_cols].isna().any().any():
            skipped_days += 1
            continue

        valid_days.append({
            "date": start.normalize(),
            "times": hidx.to_numpy(),
            "feat": g[feature_names].copy(),
            "workday": g["workday"].to_numpy(dtype=int),
            "season": g["season"].to_numpy(dtype=int),
        })

    times = []
    workday = []
    season_feat = []
    feat_arrays = {k: [] for k in feature_names}

    i = 0
    while i + 6 < len(valid_days):
        block = valid_days[i:i + 7]
        dates = [x["date"] for x in block]
        consecutive = all((dates[j + 1] - dates[j]).days == 1 for j in range(6))
        if not consecutive:
            i += 1
            continue

        times.append(np.concatenate([x["times"] for x in block]))
        workday.append(np.concatenate([x["workday"] for x in block]).astype(int))
        season_feat.append(np.concatenate([x["season"] for x in block]).astype(int))
        for k in feature_names:
            feat_arrays[k].append(
                np.concatenate([x["feat"][k].to_numpy(dtype=float) for x in block]).astype(float)
            )
        i += 7

    if len(times) == 0:
        raise ValueError(
            "No full valid consecutive weeks were created. "
            f"valid_days={len(valid_days)}, skipped_days={skipped_days}."
        )

    print(f"[INFO] valid_days={len(valid_days)} | skipped_days={skipped_days} | full_weeks={len(times)}")

    times = np.array(times, dtype=object)
    workday = np.array(workday, dtype=int)
    season_feat = np.array(season_feat, dtype=int)
    feat_arrays = {k: np.array(v, dtype=float) for k, v in feat_arrays.items()}
    return times, feat_arrays, workday, season_feat


def get_data_GEFCom2014_multi_latest(
    root: str = "/content/drive/MyDrive/M2oE2_For_Zhe/data/GEFCom2014 Data/GEFCom2014-L_V2/Load",
    timestamp_col: str = "TIMESTAMP",
    tasks: List[int] = None,
):
    """
    Latest cleaned GEFCom2014-L processing.
    Returns:
      times(object), feature_arrays(w1..w25), workday, season_feat, load, feature_names(w1..w25)
    """
    if tasks is None:
        tasks = list(range(1, 16))

    frames = []
    for i in tasks:
        csv_path = os.path.join(root, f"Task {i}", f"L{i}-train.csv")
        if not os.path.isfile(csv_path):
            print(f"[WARN] Missing file: {csv_path}")
            continue
        frames.append(_read_gefcom_task_csv(csv_path, task_id=i, timestamp_col=timestamp_col))

    if len(frames) == 0:
        raise FileNotFoundError(f"No GEFCom2014 task CSVs found under: {root}")

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["date", "__task_id__"]).groupby("date", as_index=False).last()

    required_ws = [f"w{i}" for i in range(1, 26)]
    required_cols = ["LOAD"] + required_ws
    missing = [c for c in required_cols if c not in full.columns]
    if missing:
        raise ValueError(f"Missing required GEFCom columns: {missing}")

    full = full.set_index("date").sort_index()
    full = full[required_cols]

    print(f"[INFO] GEFCom merged rows: {len(full)}")
    print(f"[INFO] LOAD missing before packing: {int(full['LOAD'].isna().sum())}")

    times, feats, workday, season_feat = _pack_hourly_df_to_weeks(full, required_cols)
    load = feats["LOAD"]
    feature_names = required_ws
    feature_arrays = [feats[w] for w in feature_names]
    return times, feature_arrays, workday.astype(int), season_feat.astype(int), load.astype(float), feature_names


def get_data_GEFCom2014_avgtemp_latest(
    root: str = "/content/drive/MyDrive/M2oE2_For_Zhe/data/GEFCom2014 Data/GEFCom2014-L_V2/Load",
    timestamp_col: str = "TIMESTAMP",
    tasks: List[int] = None,
):
    """
    Latest cleaned GEFCom2014-L processing + average of 25 weather stations.
    Returns:
      times, temp_avg, workday, season_feat, load
    """
    times, feature_arrays, workday, season_feat, load, feature_names = get_data_GEFCom2014_multi_latest(
        root=root, timestamp_col=timestamp_col, tasks=tasks
    )
    temp_stack = np.stack(feature_arrays, axis=0)   # [25, n_weeks, 168]
    temp_avg = temp_stack.mean(axis=0)              # [n_weeks, 168]
    return times, temp_avg.astype(float), workday.astype(float), season_feat.astype(float), load.astype(float)


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


@torch.no_grad()
def evaluate_rolling_first_step_for_early_stop(
    model,
    val_loader,
    device,
    quantiles=(0.1, 0.5, 0.9),
    alpha=0.1,
    deterministic_eval=True,
):
    """
    Validation metric for early stopping.
    This uses rolling first-step and deterministic z=mu by default.
    """
    model.eval()
    old_reparameterize = _enable_deterministic_reparameterize(model, deterministic_eval)

    running_mse = 0.0
    running_nll = 0.0
    running_crps = 0.0
    running_qpin = 0.0
    running_wink = 0.0
    n_samples = 0
    horizon = None

    try:
        for batch in val_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = [t.to(device) for t in batch]
            B = enc_l.size(0)

            mu_preds, logvar_preds, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
            mu_preds = mu_preds.squeeze(-1)
            logvar_preds = logvar_preds.squeeze(-1)
            tgt = tgt.squeeze(-1)

            mu_first = mu_preds[:, :, 0]
            logvar_first = logvar_preds[:, :, 0]
            tgt_first = tgt[:, :, 0]
            sigma_first = logvar_first.exp().sqrt()

            if horizon is None:
                horizon = mu_first.size(1)

            running_mse += ((mu_first - tgt_first) ** 2).mean().item() * B

            nll = 0.5 * (
                logvar_first
                + torch.log(torch.tensor(2 * np.pi, device=logvar_first.device))
                + (tgt_first - mu_first) ** 2 / (logvar_first.exp() + 1e-12)
            )
            running_nll += nll.sum().item()

            crps = crps_gaussian(mu_first, logvar_first, tgt_first)
            running_crps += crps.item() * B

            q_losses = []
            for q in quantiles:
                zq = gaussian_icdf(q, device=mu_first.device)
                yq = mu_first + sigma_first * zq
                q_losses.append(pinball_loss(tgt_first, yq, q).mean())
            running_qpin += torch.stack(q_losses).mean().item() * B

            z = gaussian_icdf(1.0 - alpha / 2.0, device=mu_first.device)
            Lb = mu_first - z * sigma_first
            Ub = mu_first + z * sigma_first
            running_wink += winkler_score(tgt_first, Lb, Ub, alpha).mean().item() * B

            n_samples += B
    finally:
        _restore_reparameterize(model, old_reparameterize)

    if n_samples == 0:
        raise ValueError("Validation loader is empty.")

    return {
        "val_mse": running_mse / n_samples,
        "val_nll": running_nll / (n_samples * horizon),
        "val_crps": running_crps / n_samples,
        "val_qpin": running_qpin / n_samples,
        "val_winkler": running_wink / n_samples,
    }


def train_model(
    model,
    train_loader,
    epochs,
    lr,
    device,
    top_k=2,
    kl_weight=0.01,
    warmup_epochs=10,
    save_path="best_model.pt",
    val_loader=None,
    patience=100,
    min_delta=0.0,
    monitor="val_crps",
    deterministic_val=True,
):
    """
    Train with validation-based early stopping.

    - Training remains stochastic VAE training.
    - Validation uses deterministic z=mu by default.
    - Best checkpoint is selected by validation metric, not training loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = float("inf")
    best_epoch = -1
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for (enc_l, enc_ext, dec_l, dec_ext, tgt) in train_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = (
                enc_l.to(device), enc_ext.to(device), dec_l.to(device), dec_ext.to(device), tgt.to(device)
            )
            optimizer.zero_grad()

            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=top_k, warmup_epochs=warmup_epochs,
            )
            nll = gaussian_nll_loss(mu_preds, logvar_preds, tgt)
            kl = kl_loss(mu_z, logvar_z)
            loss = nll + kl_weight * kl
            loss.backward()
            optimizer.step()

            running += loss.item() * enc_l.size(0)

        train_loss = running / len(train_loader.dataset)

        if val_loader is not None:
            val_metrics = evaluate_rolling_first_step_for_early_stop(
                model,
                val_loader,
                device,
                deterministic_eval=deterministic_val,
            )
            score = float(val_metrics[monitor])
        else:
            val_metrics = {}
            score = train_loss
            monitor = "train_loss"

        improved = score < (best_score - min_delta)
        if improved:
            best_score = score
            best_epoch = ep
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {ep} | {monitor} {best_score:.6f} | train loss {train_loss:.6f}")
        else:
            no_improve += 1

        if ep == 1 or ep % 5 == 0 or ep == epochs or improved:
            val_msg = ""
            if val_metrics:
                val_msg = (
                    f" | val_mse={val_metrics['val_mse']:.6f}"
                    f" val_crps={val_metrics['val_crps']:.6f}"
                    f" val_qpin={val_metrics['val_qpin']:.6f}"
                    f" val_winkler={val_metrics['val_winkler']:.6f}"
                )
            print(
                f"Epoch {ep:4d}/{epochs} | train loss: {train_loss:.6f}"
                f" | best {monitor}: {best_score:.6f} (ep {best_epoch})"
                f" | no_improve={no_improve}/{patience}{val_msg}"
            )

        if patience is not None and patience > 0 and no_improve >= patience:
            print(
                f"\n⏹ Early stopping at epoch {ep}: no validation improvement for {patience} epochs. "
                f"Best epoch {best_epoch} | best {monitor} {best_score:.6f}"
            )
            break

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
def evaluate_model(model, test_loader, loss_fn, device,
                   model_path="model.pt", reduce="first", visualize=False,
                   quantiles=(0.1, 0.5, 0.9),
                   alpha=0.1,
                   load_scale=None,
                   load_shift=None,
                   compute_raw=True,
                   compute_paper3=True,
                   deterministic_eval=True):
    """
    Default evaluation is rolling first-step.

    For every rolling window, the model outputs K steps, but this function uses
    only horizon 0:
        mu_first = mu_preds[:, :, 0]
        tgt_first = tgt[:, :, 0]

    If load_scale/load_shift are provided, raw-scale metrics are computed after
    inverse transformation:
        y_raw = y_norm * load_scale + load_shift
        sigma_raw = sigma_norm * load_scale
    """
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    old_reparameterize = _enable_deterministic_reparameterize(model, deterministic_eval)

    running_mse = 0.0
    running_nll = 0.0
    running_crps = 0.0
    running_qpin = 0.0
    running_wink = 0.0
    running_pve_abs = 0.0
    running_pve_pct = 0.0

    # Raw-scale rolling first-step metrics
    raw_enabled = compute_raw and (load_scale is not None) and (load_shift is not None)
    raw_mse_sum = 0.0
    raw_nll_sum = 0.0
    raw_crps_sum = 0.0
    raw_qpin_sum = 0.0
    raw_wink_sum = 0.0
    raw_pve_abs_sum = 0.0
    raw_pve_pct_sum = 0.0
    raw_n_points = 0

    # Raw-scale paper-style three metrics, still rolling first-step by default
    paper_pinball_sum = 0.0
    paper_winkler50_sum = 0.0
    paper_winkler90_sum = 0.0
    paper_n_points = 0
    all_qs = [q / 100.0 for q in range(1, 100)]

    for batch in test_loader:
        enc_l, enc_ext, dec_l, dec_ext, tgt = [t.to(device) for t in batch]
        B = enc_l.size(0)

        mu_preds, logvar_preds, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
        mu_preds = mu_preds.squeeze(-1)
        logvar_preds = logvar_preds.squeeze(-1)
        tgt = tgt.squeeze(-1)

        if reduce == "first":
            mu_first = mu_preds[:, :, 0]
            logvar_first = logvar_preds[:, :, 0]
            tgt_first = tgt[:, :, 0]
            sigma_first = logvar_first.exp().sqrt()

            # ---------------- normalized metrics ----------------
            running_mse += loss_fn(mu_first, tgt_first).item() * B

            p_true, _ = tgt_first.max(dim=1)
            p_pred, _ = mu_first.max(dim=1)
            diff = (p_pred - p_true).abs()
            running_pve_abs += diff.sum().item()
            running_pve_pct += (diff / (p_true + 1e-12)).sum().item()

            nll = 0.5 * (
                logvar_first +
                torch.log(torch.tensor(2 * np.pi, device=logvar_first.device)) +
                (tgt_first - mu_first) ** 2 / (logvar_first.exp() + 1e-12)
            )
            running_nll += nll.sum().item()

            crps = crps_gaussian(mu_first, logvar_first, tgt_first)
            running_crps += crps.item() * B

            q_losses = []
            for q in quantiles:
                zq = gaussian_icdf(q, device=mu_first.device)
                yq = mu_first + sigma_first * zq
                ql = pinball_loss(tgt_first, yq, q).mean()
                q_losses.append(ql)
            qpin_mean = torch.stack(q_losses).mean()
            running_qpin += qpin_mean.item() * B

            z = gaussian_icdf(1.0 - alpha / 2.0, device=mu_first.device)
            Lb = mu_first - z * sigma_first
            Ub = mu_first + z * sigma_first
            ws = winkler_score(tgt_first, Lb, Ub, alpha).mean()
            running_wink += ws.item() * B

            # ---------------- raw-scale metrics ----------------
            if raw_enabled:
                mu_raw = mu_first * load_scale + load_shift
                tgt_raw = tgt_first * load_scale + load_shift
                sigma_raw = sigma_first * load_scale
                logvar_raw = 2.0 * torch.log(sigma_raw.clamp(min=1e-8))

                raw_n_points += int(mu_raw.numel())
                raw_mse_sum += ((mu_raw - tgt_raw) ** 2).sum().item()

                raw_nll = 0.5 * (
                    logvar_raw +
                    torch.log(torch.tensor(2 * np.pi, device=logvar_raw.device)) +
                    (tgt_raw - mu_raw) ** 2 / (torch.exp(logvar_raw) + 1e-12)
                )
                raw_nll_sum += raw_nll.sum().item()

                raw_crps_sum += _crps_gaussian_from_mu_sigma(mu_raw, sigma_raw, tgt_raw).sum().item()

                raw_q_losses = []
                for q in quantiles:
                    zq = gaussian_icdf(q, device=mu_raw.device)
                    yq_raw = mu_raw + sigma_raw * zq
                    raw_q_losses.append(pinball_loss(tgt_raw, yq_raw, q).mean())
                raw_qpin_sum += torch.stack(raw_q_losses).mean().item() * B

                z_raw = gaussian_icdf(1.0 - alpha / 2.0, device=mu_raw.device)
                Lb_raw = mu_raw - z_raw * sigma_raw
                Ub_raw = mu_raw + z_raw * sigma_raw
                raw_wink_sum += winkler_score(tgt_raw, Lb_raw, Ub_raw, alpha).mean().item() * B

                p_true_raw, _ = tgt_raw.max(dim=1)
                p_pred_raw, _ = mu_raw.max(dim=1)
                raw_diff = (p_pred_raw - p_true_raw).abs()
                raw_pve_abs_sum += raw_diff.sum().item()
                raw_pve_pct_sum += (raw_diff / (p_true_raw + 1e-12)).sum().item()

                if compute_paper3:
                    # pinball_allq: 99 quantiles on raw-scale rolling first-step points
                    pb_sum_batch = 0.0
                    for q in all_qs:
                        zq = gaussian_icdf(q, device=mu_raw.device)
                        yq_raw = mu_raw + sigma_raw * zq
                        pb_sum_batch += pinball_loss(tgt_raw, yq_raw, q).sum().item()
                    paper_pinball_sum += pb_sum_batch / len(all_qs)

                    z50 = gaussian_icdf(0.75, device=mu_raw.device)
                    L50 = mu_raw - z50 * sigma_raw
                    U50 = mu_raw + z50 * sigma_raw
                    paper_winkler50_sum += winkler_score(tgt_raw, L50, U50, alpha=0.50).sum().item()

                    z90 = gaussian_icdf(0.95, device=mu_raw.device)
                    L90 = mu_raw - z90 * sigma_raw
                    U90 = mu_raw + z90 * sigma_raw
                    paper_winkler90_sum += winkler_score(tgt_raw, L90, U90, alpha=0.10).sum().item()
                    paper_n_points += int(mu_raw.numel())
        else:
            raise ValueError("reduce must be 'first' for this script")

    test_mse = running_mse / len(test_loader.dataset)
    horizon = mu_first.size(1)
    test_nll = running_nll / (len(test_loader.dataset) * horizon)
    test_crps = running_crps / len(test_loader.dataset)
    test_qpin = running_qpin / len(test_loader.dataset)
    test_wink = running_wink / len(test_loader.dataset)
    test_pve_abs = running_pve_abs / len(test_loader.dataset)
    test_pve_pct = running_pve_pct / len(test_loader.dataset)

    print("[INFO] Evaluation mode: rolling first-step")
    print(f"[INFO] Deterministic VAE eval: {deterministic_eval}")
    print(f"🧪 Normalized Test MSE         : {test_mse:.6f}")
    print(f"🧪 Normalized Test CRPS        : {test_crps:.6f}")
    print(f"🧪 Normalized Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
    print(f"🧪 Normalized Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
    print(f"🧪 Normalized Peak Value Error : {test_pve_abs:.6f} (absolute)")
    print(f"🧪 Normalized Peak Value Error%: {100.0 * test_pve_pct:.2f}% of true peak")

    raw_results = None
    paper3_results = None
    if raw_enabled:
        raw_mse = raw_mse_sum / raw_n_points
        raw_rmse = math.sqrt(raw_mse)
        raw_nll = raw_nll_sum / raw_n_points
        raw_crps = raw_crps_sum / raw_n_points
        raw_qpin = raw_qpin_sum / len(test_loader.dataset)
        raw_wink = raw_wink_sum / len(test_loader.dataset)
        raw_pve_abs = raw_pve_abs_sum / len(test_loader.dataset)
        raw_pve_pct = raw_pve_pct_sum / len(test_loader.dataset)
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
        }

        print(f"[INFO] load inverse: raw = normalized * {load_scale:.6f} + {load_shift:.6f}")
        print(f"🧪 Raw Test MSE              : {raw_mse:.6f}")
        print(f"🧪 Raw Test RMSE             : {raw_rmse:.6f}")
        print(f"🧪 Raw Test CRPS             : {raw_crps:.6f}")
        print(f"🧪 Raw Test QuantileLoss     : {raw_qpin:.6f}  (avg over q={list(quantiles)})")
        print(f"🧪 Raw Test WinklerScore     : {raw_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
        print(f"🧪 Raw Peak Value Error      : {raw_pve_abs:.6f} (absolute)")
        print(f"🧪 Raw Peak Value Error%     : {100.0 * raw_pve_pct:.2f}% of true peak")

        if compute_paper3 and paper_n_points > 0:
            pinball_allq = paper_pinball_sum / paper_n_points
            winkler_50 = paper_winkler50_sum / paper_n_points
            winkler_90 = paper_winkler90_sum / paper_n_points
            paper3_results = {
                "pinball_allq": pinball_allq,
                "winkler_50": winkler_50,
                "winkler_90": winkler_90,
                "n_points": paper_n_points,
                "mode": "rolling_first_step",
                "scale": "raw",
            }
            print("\n=== Rolling first-step paper-style metrics | raw scale ===")
            print(f"pinball_allq: {pinball_allq}")
            print(f"winkler_50  : {winkler_50}")
            print(f"winkler_90  : {winkler_90}")
            print(f"n_points    : {paper_n_points}")

    _restore_reparameterize(model, old_reparameterize)

    return {
        "normalized": {
            "MSE": test_mse,
            "NLL": test_nll,
            "CRPS": test_crps,
            "QuantileLoss": test_qpin,
            "WinklerScore": test_wink,
            "PeakValueError_abs": test_pve_abs,
            "PeakValueError_pct": 100.0 * test_pve_pct,
        },
        "raw": raw_results,
        "paper3_raw_rolling_first_step": paper3_results,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    batch_size = 16
    epochs = 500
    lr = 1e-3
    kl_weight = 0.001
    xprime_dim = 40
    hidden_dim = 64
    latent_dim = 32
    num_layers = 4
    output_len = 24   # K = 24
    top_k = 2
    warmup_ep = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_name = "GEFCom2014"
    model_name = f"M2OE2"
    model_path = f"{data_name}_{model_name}_best_model.pt"

    # Latest cleaned data + avg temperature + actually use workday and season
    times, temp_avg, workday, season_feat, load = get_data_GEFCom2014_avgtemp_latest()
    feature_dict = {
        "temp": temp_avg.astype(float),
        "workday": workday.astype(float),
        "season": season_feat.astype(float),
        "load": load.astype(float),
    }

    train_data, val_data, test_data, scalers, scaling_meta = process_seq2seq_data_gefcom_protocol_minmax(
        feature_dict=feature_dict,
        times=times,
        output_len=output_len,
        encoder_len_weeks=1,
        decoder_len_weeks=1,
        train_years=(2006, 2007, 2008, 2009),
        val_year=2010,
        test_year=2011,
        device=device,
    )
    print("[INFO] global MinMax scaling meta:")
    for k, v in scaling_meta.items():
        print(f"  {k}: {v}")

    input_dim = 1
    output_dim = 1

    n_externals = train_data['X_enc_ext'].shape[-1]
    print(f"K_ext (number of external features) = {n_externals}")
    print("[INFO] External keys used = ['temp', 'workday', 'season']")
    print("[INFO] Scaling style actually used: per-feature global MinMaxScaler (same as M2OE2_for_hr.py)")

    load_scale, load_shift = _get_load_inverse_params_from_scaler(scalers, scaling_meta)
    print(f"[INFO] load inverse transform: raw = normalized * {load_scale:.6f} + {load_shift:.6f}")

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    val_loader = make_loader(val_data, batch_size, shuffle=False)
    test_loader = make_loader(test_data, batch_size, shuffle=False)

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
    ).to(device)

    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device,
                    top_k=top_k, kl_weight=kl_weight, warmup_epochs=warmup_ep,
                    save_path=model_path, val_loader=val_loader, patience=100,
                    min_delta=0.0, monitor="val_crps", deterministic_val=True)
    else:
        print(f"[✓] Found '{model_path}', loading weights.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    import time
    time1 = time.time()
    print("\n=== Validation ===")
    evaluate_model(model, val_loader, nn.MSELoss(), device, model_path=model_path, visualize=False,
                   load_scale=load_scale, load_shift=load_shift, compute_raw=True, compute_paper3=True, deterministic_eval=True)
    print("\n=== Test ===")
    evaluate_model(model, test_loader, nn.MSELoss(), device, model_path=model_path, visualize=False,
                   load_scale=load_scale, load_shift=load_shift, compute_raw=True, compute_paper3=True, deterministic_eval=True)
    time2 = time.time()
    print("time", time2 - time1)
