# peak_metrics.py
# ------------------------------------------------------------------------------
# NOTE: This module serves as an auxiliary utility for metric testing and 
# verification. It is not a core component of the primary modeling framework 
# and is intended solely for supplementary evaluation.
# ------------------------------------------------------------------------------

import numpy as np
from typing import Dict, Tuple, List, Optional

# ==========================
# Basic Utilities
# ==========================
def _rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    """
    Compute rolling mean of length k (valid mode).
    """
    if k <= 1:
        return x.copy()
    k = int(k)
    w = np.ones(k, dtype=float) / k
    return np.convolve(x.astype(float), w, mode="valid")

def _argmax_first(x: np.ndarray) -> int:
    idx = int(np.argmax(x))
    return idx

def _split_week_into_days(y_week: np.ndarray, points_per_day: Optional[int] = None) -> List[np.ndarray]:
    """
    Split a weekly sequence into 7 days.
    Defaults to 24 points/day given a standard 168 points/week input.
    """
    T = len(y_week)
    if points_per_day is None:
        # Attempt automatic inference (common: 168->24, 672->96, etc.)
        for cand in (24, 48, 96, 120, 144):  # 1h / 30min / 15min / 12min / 10min
            if T % (7*cand) == 0:
                points_per_day = cand
                break
        if points_per_day is None:
            # Fallback: Use nearest integer
            points_per_day = T // 7
            if 7 * points_per_day != T:
                raise ValueError(f"Cannot split week length {T} into 7 equal days.")
    days = []
    for d in range(7):
        days.append(y_week[d*points_per_day:(d+1)*points_per_day])
    return days

def _interval_iou(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two intervals [a0, a1), [b0, b1).
    Unit: Minutes (must be consistent).
    """
    a0,a1 = a; b0,b1 = b
    inter = max(0, min(a1,b1) - max(a0,b0))
    uni   = (a1-a0) + (b1-b0) - inter
    return inter/uni if uni>0 else 0.0

def _span_to_idx(span, dt_min: int):
    """
    Map time interval (start_min, end_min) in minutes to index range [s_idx:e_idx).
    """
    if span is None:
        return None
    s = max(0, int(round(span[0] / dt_min)))
    e = max(s, int(round(span[1] / dt_min)))
    return (s, e)

def _mse_pair(a: np.ndarray, b: np.ndarray) -> float:
    L = min(len(a), len(b))
    if L <= 0:
        return float("nan")
    return float(np.mean((a[:L] - b[:L]) ** 2))

# ==========================
# Peak Values and Periods (Compatible with ERCOT/PJM methodologies)
# ==========================
def daily_peak_value(y: np.ndarray, dt_min: int, tau_min: Optional[int] = None) -> Tuple[float,int,int,np.ndarray]:
    """
    Calculate Daily Peak Value (DPV) based on demand (max window average).
    
    Returns: 
        (DPV, t_star_min, k, y_tau)
        t_star_min is the 'start minute' of the peak window.
    """
    tau = tau_min or dt_min         # Use current sampling interval if no finer granularity is provided
    k = max(1, int(round(tau / dt_min)))
    y_tau = _rolling_mean(y, k)     # Length T - k + 1
    idx = _argmax_first(y_tau)
    dpv = float(y_tau[idx])
    t_star_min = idx * dt_min
    return dpv, t_star_min, k, y_tau

def max_window_peak_period(y: np.ndarray, dt_min: int, W_min: Optional[int] = None) -> Tuple[int,int]:
    """
    Identify the continuous window of length W_min with the highest average value within a day.
    
    Returns:
        (start_min, end_min)
    """
    W = W_min or dt_min
    m = max(1, int(round(W / dt_min)))
    y_W = _rolling_mean(y, m)
    idx = _argmax_first(y_W)
    t0_min = idx * dt_min
    t1_min = t0_min + W
    return t0_min, t1_min

def threshold_peak_period(y_tau: np.ndarray, dt_min: int, dpv: float, alpha: float = 0.9) -> Optional[Tuple[int,int]]:
    """
    Identify the longest continuous interval on the 'tau-averaged' curve (y_tau) 
    where values exceed alpha * dpv.
    
    Returns:
        (start_min, end_min); None if not found.
    """
    mask = (y_tau >= alpha * dpv).astype(int)
    best = (None, 0)   # (interval, length)
    s = None
    for i in range(len(mask)+1):
        cur = mask[i] if i < len(mask) else 0
        prev = mask[i-1] if i>0 else 0
        if cur==1 and (i==0 or prev==0):
            s = i
        if (cur==0 and prev==1) and s is not None:
            span = (s*dt_min, i*dt_min)
            L = span[1]-span[0]
            if L > best[1]:
                best = (span, L)
            s = None
    return best[0]

# ==========================
# Metrics Aggregation
# ==========================
def peak_metrics_day(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dt_min: int = 60,
    tau_min: Optional[int] = None,
    W_min: Optional[int]  = None,
    alpha: float = 0.90
) -> Dict[str, float]:
    """
    Compute peak value, timing, period, and error metrics for a 'single day'.
    
    Returns a dictionary containing:
      - PVAE: Peak Value Absolute Error
      - PVPE: Peak Value Percentage Error
      - PTE_min: Peak Time Error (minutes)
      - PeakWindowIoU: IoU of max-window peak period
      - ThrWindowIoU: IoU of threshold-based peak period (NaN if either is None)
      - Plus DPV, timing, and intervals for both ground truth and prediction.
    """
    # 1) Peak Value (Demand Methodology)
    dpv_t, tstar_t, k_t, ytau_t = daily_peak_value(y_true, dt_min, tau_min)
    dpv_p, tstar_p, k_p, ytau_p = daily_peak_value(y_pred, dt_min, tau_min)

    # 2) Peak Period (A: Max Window)
    win_t = max_window_peak_period(y_true, dt_min, W_min)
    win_p = max_window_peak_period(y_pred, dt_min, W_min)
    iou_win = _interval_iou(win_t, win_p)

    # 3) Peak Period (B: Threshold Method)
    thr_t = threshold_peak_period(ytau_t, dt_min, dpv_t, alpha)
    thr_p = threshold_peak_period(ytau_p, dt_min, dpv_p, alpha)

    thr_idx_t = _span_to_idx(thr_t, dt_min)
    thr_idx_p = _span_to_idx(thr_p, dt_min)

    mse_truewin = float("nan")
    mse_predwin = float("nan")
    mse_union   = float("nan")
    mse_inter   = float("nan")

    if thr_idx_t is not None:
        s, e = thr_idx_t
        mse_truewin = _mse_pair(y_pred[s:e], y_true[s:e])

    if thr_idx_p is not None:
        s, e = thr_idx_p
        mse_predwin = _mse_pair(y_pred[s:e], y_true[s:e])

    if (thr_idx_t is not None) and (thr_idx_p is not None):
        # Union
        s_u = min(thr_idx_t[0], thr_idx_p[0]); e_u = max(thr_idx_t[1], thr_idx_p[1])
        mse_union = _mse_pair(y_pred[s_u:e_u], y_true[s_u:e_u])
        # Intersection
        s_i = max(thr_idx_t[0], thr_idx_p[0]); e_i = min(thr_idx_t[1], thr_idx_p[1])
        if e_i > s_i:
            mse_inter = _mse_pair(y_pred[s_i:e_i], y_true[s_i:e_i])

    if thr_t is None or thr_p is None:
        iou_thr = float("nan")
    else:
        iou_thr = _interval_iou(thr_t, thr_p)

    # 4) Error Metrics
    pvae = abs(dpv_p - dpv_t)
    pvpe = pvae / max(dpv_t, 1e-6)
    pte  = abs(tstar_p - tstar_t)

    return dict(
        PVAE=pvae,
        PVPE=pvpe,
        PTE_min=pte,
        PeakWindowIoU=iou_win,
        ThrWindowIoU=iou_thr,
        PeakValue_true=dpv_t,
        PeakValue_pred=dpv_p,
        PeakTime_true_min=tstar_t,
        PeakTime_pred_min=tstar_p,
        PeakWin_true_min_start=win_t[0], PeakWin_true_min_end=win_t[1],
        PeakWin_pred_min_start=win_p[0], PeakWin_pred_min_end=win_p[1],
        k_tau_true=k_t, k_tau_pred=k_p,
        ThrPeakMSE_truewin = mse_truewin,
        ThrPeakMSE_predwin = mse_predwin,
        ThrPeakMSE_union   = mse_union,
        ThrPeakMSE_inter   = mse_inter,
    )

def summarize_metrics(metrics_list: List[Dict[str,float]]) -> Dict[str,float]:
    """
    Aggregate 7-day metrics using mean/median (expandable as needed).
    """
    import math
    keys = [
        "PVAE","PVPE","PTE_min","PeakWindowIoU","ThrWindowIoU",
        "ThrPeakMSE_truewin","ThrPeakMSE_predwin","ThrPeakMSE_union","ThrPeakMSE_inter"
    ]
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if (not isinstance(m[k], float) or not math.isnan(m[k]))]
        if len(vals)==0:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_median"] = float("nan")
        else:
            agg[f"{k}_mean"]   = float(np.mean(vals))
            agg[f"{k}_median"] = float(np.median(vals))
    return agg

# ==========================
# Weekly Sequence Interface (Aligned with current output)
# ==========================
def peak_metrics_week(
    y_true_week: np.ndarray,
    y_pred_week: np.ndarray,
    *,
    dt_min: int = 60,
    tau_min: Optional[int] = None,
    W_min: Optional[int]  = None,
    alpha: float = 0.90,
    points_per_day: Optional[int] = None
) -> Dict[str,object]:
    """
    For a 'weekly' sequence (length = 7 * points_per_day):
      - Compute metrics daily, returning a list of day_metrics (len=7)
      - Additionally compute for the whole week (treating the week as a 'long day') -> week_level
      - Return comprehensive summary (mean/median over 7 days)
    """
    days_t = _split_week_into_days(y_true_week, points_per_day)
    days_p = _split_week_into_days(y_pred_week, points_per_day)

    day_metrics = []
    for d in range(7):
        m = peak_metrics_day(
            days_t[d], days_p[d],
            dt_min=dt_min, tau_min=tau_min, W_min=W_min, alpha=alpha
        )
        m["day_index"] = int(d)
        day_metrics.append(m)

    summary = summarize_metrics(day_metrics)

    # Treat the week as a 'single day' for calculation (facilitates weekly comparison)
    week_level = peak_metrics_day(
        y_true_week, y_pred_week,
        dt_min=dt_min, tau_min=tau_min, W_min=W_min, alpha=alpha
    )

    return dict(
        day_metrics=day_metrics,
        week_metrics=week_level,
        summary=summary
    )

# ==========================
# Recommended Default Parameters (Texas/ERCOT)
# ==========================
def ercot_defaults(dt_min: int) -> Dict[str,object]:
    """
    ERCOT 4CP demand methodology uses a 15-minute interval.
    
    If 15min data is available:
        tau_min=15, W_min=15
    If currently using 60min (hourly):
        Approximate with hours (tau_min=60, W_min=60), update when 15min data becomes available.
    """
    if dt_min <= 15:
        return dict(dt_min=dt_min, tau_min=15, W_min=15, alpha=0.90)
    else:
        return dict(dt_min=dt_min, tau_min=60, W_min=60, alpha=0.90)