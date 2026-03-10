"""
This file will draw the two time period of the dataset, the "temperature sudden change" (Top1) and the "load sudden change" (Top1).
Goal
----
1) Find the week with the largest "temperature sudden change" (Top1) for a target XFMR.
2) Find the week with the largest "load sudden change" (Top1) for a target XFMR.
3) For each Top1 week, plot (base-only) with your existing style:
   - History (black) + True (red) + Base pred (blue) + Base ±1σ
   - Temperature on secondary axis

This version ONLY changes plotting/output so it will NOT overwrite previous figures.

Inputs needed:
- CSV_PATH         : XFMR_forecast_data_20251113.csv
- XLSX_PATH        : exported XLSX from evaluate_model (base-only or base+)
- SCALER_META_JSON : scaler meta json (load_min/load_max)

Output:
- a NEW unique output folder under OUT_DIR_BASE
- figure filenames include the XLSX base name to avoid collisions
- selection_summary.json saved for slides/repro
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# Config  (EDIT THESE)
# =========================
CSV_PATH = "XFMR_forecast_data_20251113.csv"

# >>> point this to your current model export <<<
# e.g. "vae_base_only_v5_v1temp_oracle_XFMR125916888.xlsx"
XLSX_PATH = "vae_base_only_v5_XFMR125916888.xlsx"

# >>> point this to your current scaler meta <<<
# e.g. "vae_base_scaler_meta_v5_v1temp_oracle.json"
SCALER_META_JSON = "vae_base_scaler_meta_v5.json"

TARGET_XFMR = data1
ENCODER_LEN_WEEKS = 1

BASE_MODEL_KEYS = ["VAE_BASE_V5", "BASE_V5", "BASE"]

# ---- output: never overwrite ----
OUT_DIR_BASE = "sudden_change_top1_plots_v5"  # base folder
SAVE_FIG = True
SHOW_FIG = True

LOAD_UNIT = "kW"
TEMP_UNIT = "°F"

# ---- shock selection params ----
Q = 0.95
TOP_N_SEARCH = 10
USE_BOUNDARY = True
USE_WITHIN = True


# =========================
# Helpers (I/O + denorm)
# =========================
def resolve_xlsx_path(xlsx_path: str) -> str:
    if os.path.exists(xlsx_path):
        return xlsx_path

    cands = glob.glob("*.xlsx")
    if not cands:
        raise FileNotFoundError(
            f"[ERR] XLSX not found: {xlsx_path}\n"
            f"[ERR] Also no .xlsx files found in current directory."
        )

    def score(p):
        name = p.lower()
        s = 0
        if "v5" in name: s += 10
        if "v1temp" in name: s += 6
        if "oracle" in name: s += 5
        if "peak" in name: s += 4
        if "comparison" in name: s += 3
        if "base" in name: s += 2
        if "lora" in name: s += 1
        return s

    best = sorted(cands, key=score, reverse=True)[0]
    print(f"[WARN] XLSX_PATH not found: {xlsx_path}")
    print(f"[INFO] Auto-selected XLSX: {best}")
    return best


def load_scaler_meta(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERR] scaler meta JSON not found: {path}\n"
            f"Hint: Ensure '{os.path.basename(path)}' is in the directory."
        )
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    need = ["load_min", "load_max"]
    miss = [k for k in need if k not in meta]
    if miss:
        raise ValueError(f"scaler_meta missing keys: {miss}, got keys={list(meta.keys())}")
    return meta


def denorm_minmax(x, vmin, vmax):
    return x * (vmax - vmin) + vmin


def pick_model_name_by_keys(df_xlsx: pd.DataFrame, keys):
    if "model_name" not in df_xlsx.columns:
        return None
    names = sorted(df_xlsx["model_name"].dropna().unique().tolist())
    for k in keys:
        hits = [n for n in names if k in str(n)]
        if hits:
            return hits[0]
    return None


def extract_one_sample(df_xlsx: pd.DataFrame, sample_index: int, model_name: str):
    if model_name is None:
        return None

    if "sample_index" not in df_xlsx.columns or "model_name" not in df_xlsx.columns:
        raise ValueError("XLSX must contain columns: sample_index, model_name")

    mask = (df_xlsx["sample_index"] == sample_index) & (df_xlsx["model_name"] == model_name)

    if "feature_name" in df_xlsx.columns:
        mask = mask & (df_xlsx["feature_name"] == "load")

    dm = df_xlsx[mask].copy()
    if dm.empty:
        return None

    def get(vtype: str):
        x = dm[dm["value_type"] == vtype].sort_values("time_step")
        if x.empty:
            return np.array([]), np.array([])
        return x["time_step"].to_numpy(), x["value"].to_numpy()

    x_h, y_h = get("history")
    x_t, y_t = get("true")
    x_p, y_p = get("pred_mean")
    x_s, y_s = get("pred_std")

    return {"x_h": x_h, "hist": y_h, "x_t": x_t, "true": y_t, "x_p": x_p, "pred": y_p, "std": y_s}


def denorm_pack_load(pack, meta):
    if pack is None:
        return None
    pack = dict(pack)
    load_min, load_max = meta["load_min"], meta["load_max"]
    load_scale = (load_max - load_min)

    if len(pack["hist"]) > 0: pack["hist"] = denorm_minmax(pack["hist"], load_min, load_max)
    if len(pack["true"]) > 0: pack["true"] = denorm_minmax(pack["true"], load_min, load_max)
    if len(pack["pred"]) > 0: pack["pred"] = denorm_minmax(pack["pred"], load_min, load_max)
    if len(pack["std"])  > 0: pack["std"]  = pack["std"] * load_scale
    return pack


# =========================
# CSV parsing + week indexing
# =========================
def _guess_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_hourly_df(csv_path: str, xfmr: int):
    df = pd.read_csv(csv_path)
    df = df[df["XFMR"] == xfmr].copy()
    if df.empty:
        raise ValueError(f"No rows for XFMR={xfmr} in {csv_path}")

    df["DATE1"] = pd.to_datetime(df["DATE1"])
    df["ts"] = df["DATE1"] + pd.to_timedelta(df["HOUR1"].astype(int), unit="h")

    temp_col = _guess_col(df, ("TEMPERATURE", "TEMP", "Temperature"))
    if temp_col is None:
        raise KeyError(f"[ERR] Cannot find temperature column. Columns={df.columns.tolist()}")

    load_col = _guess_col(df, (
        "LOAD", "Load", "LOAD_KW", "LOAD_AVG", "KW", "kW", "LOAD1", "LOAD_VALUE",
        "LOAD_MW", "MW", "P", "POWER", "P_KW"
    ))
    if load_col is None:
        for c in df.columns:
            if isinstance(c, str) and ("load" in c.lower()):
                load_col = c
                break

    agg_cols = {temp_col: "mean"}
    if load_col is not None:
        agg_cols[load_col] = "mean"

    hourly = df.groupby("ts", as_index=False).agg(agg_cols).sort_values("ts").reset_index(drop=True)

    t0 = hourly["ts"].min()
    hourly["week_idx"] = ((hourly["ts"] - t0).dt.total_seconds() // (3600 * 168)).astype(int)

    week_start = hourly.groupby("week_idx")["ts"].min().rename("week_start_ts").reset_index()

    print(f"[CSV] XFMR {xfmr} time range: {hourly['ts'].min()} -> {hourly['ts'].max()}")
    print(f"[CSV] temp_col={temp_col} | load_col={load_col if load_col is not None else '(NOT FOUND)'}")
    return hourly, week_start, temp_col, load_col


# =========================
# Sudden-week finder
# =========================
def find_top_sudden_weeks_from_hourly(
    hourly: pd.DataFrame,
    value_col: str,
    *,
    q: float = 0.95,
    top_n: int = 10,
    use_boundary: bool = True,
    use_within: bool = True,
):
    d = hourly[["ts", "week_idx", value_col]].copy()
    d["dV"] = d[value_col].diff().abs()

    rows = []
    for w, wk in d.groupby("week_idx", sort=True):
        if len(wk) < 2:
            continue

        dV_within = float(wk["dV"].max()) if use_within else 0.0

        if use_boundary and w > 0:
            first_pos = wk.index.min()
            dV_boundary = float(d.loc[first_pos, "dV"])
        else:
            dV_boundary = 0.0

        dV_metric = max(dV_within, dV_boundary)

        rows.append({
            "week_idx": int(w),
            "week_start_ts": wk["ts"].min(),
            "dV_metric": dV_metric,
            "dV_within": dV_within,
            "dV_boundary": dV_boundary,
        })

    wkdf = pd.DataFrame(rows)
    if wkdf.empty:
        return wkdf

    thr = wkdf["dV_metric"].quantile(q)
    cand = wkdf[wkdf["dV_metric"] >= thr].copy()
    cand = cand.sort_values("dV_metric", ascending=False).head(top_n).reset_index(drop=True)

    print(f"[SuddenWeeks] col={value_col} q={q} thr={thr:.3f} candidates={len(cand)} (top_n={top_n})")
    return cand


# =========================
# Load shock fallback via XLSX
# =========================
def compute_load_shock_from_xlsx(df_xlsx: pd.DataFrame, meta: dict, model_base: str):
    if model_base is None:
        raise ValueError("model_base is None; cannot compute load shock from XLSX.")

    dfb = df_xlsx[df_xlsx["model_name"] == model_base].copy()
    if dfb.empty:
        raise ValueError(f"No rows for model_base='{model_base}' in XLSX.")

    if "feature_name" in dfb.columns:
        dfb = dfb[dfb["feature_name"] == "load"].copy()

    sample_ids = sorted(dfb["sample_index"].dropna().unique().tolist())

    out = []
    for sid in sample_ids:
        pack = extract_one_sample(df_xlsx, int(sid), model_base)
        if pack is None:
            continue
        pack = denorm_pack_load(pack, meta)

        hist = pack["hist"]
        true = pack["true"]
        if len(true) < 2:
            continue

        within = float(np.max(np.abs(np.diff(true))))
        boundary = float(abs(true[0] - hist[-1])) if (len(hist) > 0) else 0.0
        metric = max(within, boundary)
        out.append({"sample_index": int(sid), "dL_metric": metric, "dL_within": within, "dL_boundary": boundary})

    return pd.DataFrame(out).sort_values("dL_metric", ascending=False).reset_index(drop=True)


# =========================
# Plot utilities
# =========================
def load_temp_from_csv(csv_path: str, xfmr: int, start_ts: pd.Timestamp, periods: int):
    df = pd.read_csv(csv_path)
    df = df[df["XFMR"] == xfmr].copy()
    if df.empty:
        raise ValueError(f"No rows for XFMR={xfmr} in {csv_path}")

    df["DATE1"] = pd.to_datetime(df["DATE1"])
    df["ts"] = df["DATE1"] + pd.to_timedelta(df["HOUR1"].astype(int), unit="h")

    temp_col = _guess_col(df, ("TEMPERATURE", "TEMP", "Temperature"))
    if temp_col is None:
        raise KeyError(f"[ERR] Cannot find temperature column. Columns={df.columns.tolist()}")

    temp_by_ts = df.groupby("ts", as_index=True)[temp_col].mean().sort_index()

    idx = pd.date_range(start=start_ts, periods=periods, freq="h")
    out = temp_by_ts.reindex(idx).ffill().bfill()
    return idx, out.to_numpy()


def make_unique_outdir(base_dir: str, xlsx_path: str):
    """
    Create a unique output folder to avoid overwriting.
    Includes xlsx stem + timestamp.
    """
    stem = os.path.splitext(os.path.basename(xlsx_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"{stem}__{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_one_week(
    *,
    df_xlsx: pd.DataFrame,
    sample_index: int,
    decoder_week_start_ts: pd.Timestamp,
    tag: str,
    out_dir: str,
    meta: dict,
    load_unit: str,
    temp_unit: str,
    model_base: str,
    csv_path: str,
    xfmr: int,
    xlsx_tag: str,
    save_fig: bool = True,
    show_fig: bool = True,
):
    base = extract_one_sample(df_xlsx, sample_index, model_base) if model_base else None
    if base is None:
        raise ValueError(f"sample_index={sample_index} not found for model_base={model_base}")

    base = denorm_pack_load(base, meta)

    def get_max_step(p):
        v = []
        if len(p["x_h"]): v.append(p["x_h"].max())
        if len(p["x_t"]): v.append(p["x_t"].max())
        if len(p["x_p"]): v.append(p["x_p"].max())
        return int(max(v)) if v else 0

    max_step = get_max_step(base)
    total_len = max_step + 1
    if total_len <= 1:
        print(f"[WARN] total_len={total_len} too small. Skipping.")
        return None

    history_start = decoder_week_start_ts - pd.Timedelta(hours=168)
    full_time = pd.date_range(start=history_start, periods=total_len, freq="h")

    def to_dt(x_steps):
        if len(x_steps) == 0:
            return []
        x_steps = x_steps.astype(int)
        x_steps = np.clip(x_steps, 0, total_len - 1)
        return full_time[x_steps]

    temp_time, temp_vals = load_temp_from_csv(csv_path, xfmr, history_start, total_len)

    plt.figure(figsize=(12, 3.6))
    ax = plt.gca()

    # History & True
    if len(base["x_h"]) > 0:
        ax.plot(to_dt(base["x_h"]), base["hist"], color="black", linewidth=1.5, label="History")
    if len(base["x_t"]) > 0:
        ax.plot(to_dt(base["x_t"]), base["true"], linestyle="--", color="red", linewidth=1.5, label="True")

    # Base pred
    if len(base["x_p"]) > 0:
        ax.plot(to_dt(base["x_p"]), base["pred"], color="blue", alpha=0.9, linewidth=1.5, label="Base (mean)")
        if len(base["std"]) == len(base["pred"]) and len(base["std"]) > 0:
            ax.fill_between(
                to_dt(base["x_p"]),
                base["pred"] - base["std"],
                base["pred"] + base["std"],
                color="blue", alpha=0.15, label="Base (±1σ)"
            )

    # split marker
    if total_len > 168:
        ax.axvline(full_time[167], color="grey", linestyle="--", alpha=0.5)

    ax.set_ylabel(f"Load ({load_unit})")
    ax.set_xlabel("Date")
    ax.set_title(
        f"{tag} | XFMR={xfmr} | Week={pd.to_datetime(decoder_week_start_ts).date()} | sample={sample_index}\n"
        f"XLSX={xlsx_tag} | model={model_base}"
    )

    # Temp on secondary axis
    ax2 = ax.twinx()
    if len(temp_time) > 168:
        ax2.plot(temp_time[:168], temp_vals[:168], linestyle=":", linewidth=1.0, alpha=0.5, label="Temp (hist)")
        ax2.plot(temp_time[168:], temp_vals[168:], linestyle=":", linewidth=1.0, alpha=0.5, label="Temp (fore)")
    else:
        ax2.plot(temp_time, temp_vals, linestyle=":", linewidth=1.0, alpha=0.5, label="Temp")

    ax2.set_ylabel(f"Temperature ({temp_unit})")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    out_png = None
    if save_fig:
        week_str = str(pd.to_datetime(decoder_week_start_ts).date())
        # filename includes xlsx tag so it never collides with other runs
        safe_tag = tag.replace(" ", "_")
        out_png = os.path.join(
            out_dir,
            f"{xlsx_tag}__{safe_tag}__xfmr{xfmr}__week{week_str}__sample{sample_index}__base.png"
        )
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[✓] Saved: {out_png}")

    if show_fig:
        plt.show()
    else:
        plt.close()

    return out_png


# =========================
# Candidate selection that guarantees sample exists
# =========================
def pick_first_valid_candidate_week(
    cand_weeks: pd.DataFrame,
    df_xlsx: pd.DataFrame,
    model_base: str,
    encoder_len_weeks: int,
):
    if cand_weeks is None or cand_weeks.empty:
        return None

    for i in range(min(len(cand_weeks), TOP_N_SEARCH)):
        w = int(cand_weeks.loc[i, "week_idx"])
        ts = pd.to_datetime(cand_weeks.loc[i, "week_start_ts"])
        sample_idx = int(w - encoder_len_weeks)
        if sample_idx < 0:
            continue
        pack = extract_one_sample(df_xlsx, sample_idx, model_base)
        if pack is not None:
            return w, ts, sample_idx, float(cand_weeks.loc[i, "dV_metric"])
    return None


# =========================
# Main
# =========================
def main():
    # Load XLSX + meta
    xlsx_path = resolve_xlsx_path(XLSX_PATH)
    meta = load_scaler_meta(SCALER_META_JSON)

    df_xlsx = pd.read_excel(xlsx_path)
    model_base = pick_model_name_by_keys(df_xlsx, BASE_MODEL_KEYS)
    print(f"[INFO] XLSX: {xlsx_path}")
    print(f"[INFO] model_base picked: {model_base}")
    if model_base is None:
        raise ValueError("Cannot find base model_name in XLSX. Please adjust BASE_MODEL_KEYS.")

    # Make unique output dir (never overwrite)
    out_dir = make_unique_outdir(OUT_DIR_BASE, xlsx_path)
    xlsx_tag = os.path.splitext(os.path.basename(xlsx_path))[0]
    print(f"[INFO] Output folder: {out_dir}")

    # Load CSV hourly + week map
    hourly, week_start_map, temp_col, load_col = load_hourly_df(CSV_PATH, TARGET_XFMR)

    summary = {
        "xfmr": int(TARGET_XFMR),
        "xlsx": os.path.basename(xlsx_path),
        "scaler_meta": os.path.basename(SCALER_META_JSON),
        "csv": os.path.basename(CSV_PATH),
        "model_base": model_base,
        "params": {
            "Q": Q,
            "TOP_N_SEARCH": TOP_N_SEARCH,
            "USE_BOUNDARY": USE_BOUNDARY,
            "USE_WITHIN": USE_WITHIN,
            "ENCODER_LEN_WEEKS": ENCODER_LEN_WEEKS,
        },
        "results": {}
    }

    # -------- (1) TEMP shock Top1 --------
    cand_temp = find_top_sudden_weeks_from_hourly(
        hourly, temp_col, q=Q, top_n=TOP_N_SEARCH,
        use_boundary=USE_BOUNDARY, use_within=USE_WITHIN,
    )
    pick_temp = pick_first_valid_candidate_week(cand_temp, df_xlsx, model_base, ENCODER_LEN_WEEKS)
    if pick_temp is None:
        print("[WARN] No valid TEMP shock week found.")
        summary["results"]["temp_top1"] = None
    else:
        wT, tsT, sT, metricT = pick_temp
        print(f"[TOP1 TEMP] week_idx={wT}, week_start={tsT}, sample_index={sT}, metric={metricT:.3f}")
        pngT = plot_one_week(
            df_xlsx=df_xlsx,
            sample_index=sT,
            decoder_week_start_ts=tsT,
            tag=f"TEMP_SHOCK_TOP1_w{wT}",
            out_dir=out_dir,
            meta=meta,
            load_unit=LOAD_UNIT,
            temp_unit=TEMP_UNIT,
            model_base=model_base,
            csv_path=CSV_PATH,
            xfmr=TARGET_XFMR,
            xlsx_tag=xlsx_tag,
            save_fig=SAVE_FIG,
            show_fig=SHOW_FIG,
        )
        summary["results"]["temp_top1"] = {
            "week_idx": int(wT),
            "week_start_ts": str(pd.to_datetime(tsT)),
            "sample_index": int(sT),
            "shock_metric": float(metricT),
            "png": os.path.basename(pngT) if pngT else None
        }

    # -------- (2) LOAD shock Top1 --------
    if load_col is not None:
        cand_load = find_top_sudden_weeks_from_hourly(
            hourly, load_col, q=Q, top_n=TOP_N_SEARCH,
            use_boundary=USE_BOUNDARY, use_within=USE_WITHIN,
        )
        pick_load = pick_first_valid_candidate_week(cand_load, df_xlsx, model_base, ENCODER_LEN_WEEKS)
        if pick_load is None:
            print("[WARN] No valid LOAD shock week found.")
            summary["results"]["load_top1"] = None
        else:
            wL, tsL, sL, metricL = pick_load
            print(f"[TOP1 LOAD] week_idx={wL}, week_start={tsL}, sample_index={sL}, metric={metricL:.3f}")
            pngL = plot_one_week(
                df_xlsx=df_xlsx,
                sample_index=sL,
                decoder_week_start_ts=tsL,
                tag=f"LOAD_SHOCK_TOP1_w{wL}",
                out_dir=out_dir,
                meta=meta,
                load_unit=LOAD_UNIT,
                temp_unit=TEMP_UNIT,
                model_base=model_base,
                csv_path=CSV_PATH,
                xfmr=TARGET_XFMR,
                xlsx_tag=xlsx_tag,
                save_fig=SAVE_FIG,
                show_fig=SHOW_FIG,
            )
            summary["results"]["load_top1"] = {
                "week_idx": int(wL),
                "week_start_ts": str(pd.to_datetime(tsL)),
                "sample_index": int(sL),
                "shock_metric": float(metricL),
                "png": os.path.basename(pngL) if pngL else None
            }
    else:
        print("[INFO] CSV has no load column. Falling back to XLSX true/history for LOAD shock...")
        df_shock = compute_load_shock_from_xlsx(df_xlsx, meta, model_base)
        if df_shock.empty:
            print("[WARN] XLSX-based load shock list is empty.")
            summary["results"]["load_top1"] = None
        else:
            picked = None
            for i in range(min(len(df_shock), TOP_N_SEARCH)):
                sidx = int(df_shock.loc[i, "sample_index"])
                widx = int(sidx + ENCODER_LEN_WEEKS)
                hit = week_start_map[week_start_map["week_idx"] == widx]
                if hit.empty:
                    continue
                tsL = pd.to_datetime(hit.iloc[0]["week_start_ts"])
                pack = extract_one_sample(df_xlsx, sidx, model_base)
                if pack is None:
                    continue
                picked = (widx, tsL, sidx, float(df_shock.loc[i, "dL_metric"]))
                break

            if picked is None:
                print("[WARN] Could not map any top XLSX load-shock sample to CSV week_start_ts.")
                summary["results"]["load_top1"] = None
            else:
                wL, tsL, sL, metric = picked
                print(f"[TOP1 LOAD (XLSX)] week_idx={wL}, week_start={tsL}, sample_index={sL}, metric={metric:.3f}")
                pngL = plot_one_week(
                    df_xlsx=df_xlsx,
                    sample_index=sL,
                    decoder_week_start_ts=tsL,
                    tag=f"LOAD_SHOCK_TOP1_XLSX_w{wL}",
                    out_dir=out_dir,
                    meta=meta,
                    load_unit=LOAD_UNIT,
                    temp_unit=TEMP_UNIT,
                    model_base=model_base,
                    csv_path=CSV_PATH,
                    xfmr=TARGET_XFMR,
                    xlsx_tag=xlsx_tag,
                    save_fig=SAVE_FIG,
                    show_fig=SHOW_FIG,
                )
                summary["results"]["load_top1"] = {
                    "week_idx": int(wL),
                    "week_start_ts": str(pd.to_datetime(tsL)),
                    "sample_index": int(sL),
                    "shock_metric": float(metric),
                    "png": os.path.basename(pngL) if pngL else None
                }

    # Save summary JSON (for slides / reproducibility)
    sum_path = os.path.join(out_dir, "selection_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Saved summary: {sum_path}")


if __name__ == "__main__":
    main()
