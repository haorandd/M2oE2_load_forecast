# ============================================================
# V5 BASE TRAINING (v1_temp): add oracle temp_fc_* features
#   - NO LoRA
#   - temp_fc_* computed from temp_true (upper-bound / oracle)
#   - Saves: base ckpt, scaler meta json, training config json, eval xlsx
# ============================================================

import os, random, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.distributions.normal import Normal
from torch.optim import AdamW

from data_utils_v2 import *
from data_utils_v2 import reconstruct_sequence
from model_v1 import *
from peak_metrics import peak_metrics_week, ercot_defaults


# ====================== Common helpers ======================

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


# ============================================================
# v1_temp NEW: compute oracle forward-24h temperature summaries
# ============================================================

def compute_forward_24h_temp_stats(temp_1d: np.ndarray, horizon: int = 24):
    """
    Oracle version: uses TRUE future temperature to create features.
    For each time t:
        mean24(t) = mean(temp[t:t+24])
        max24(t)  = max(...)
        min24(t)  = min(...)
        ramp24(t) = max - min

    Returns 4 arrays, same length as temp_1d.
    """
    s = pd.Series(np.asarray(temp_1d, dtype=float))
    s_rev = s.iloc[::-1]
    mean24 = s_rev.rolling(horizon, min_periods=1).mean().iloc[::-1].to_numpy()
    max24  = s_rev.rolling(horizon, min_periods=1).max().iloc[::-1].to_numpy()
    min24  = s_rev.rolling(horizon, min_periods=1).min().iloc[::-1].to_numpy()
    ramp24 = (max24 - min24)
    return mean24, max24, min24, ramp24


# ====================== Seq2Seq processing (fit scalers) ======================

def process_seq2seq_data(
        feature_dict,
        *,
        train_ratio=0.7,
        output_len=24,
        encoder_len_weeks=1,
        decoder_len_weeks=1,
        num_in_week=168,
        device=None):

    processed, scalers = {}, {}
    for k, arr in feature_dict.items():
        if arr.size == 0:
            raise ValueError(f"feature '{k}' is empty.")
        vec = np.asarray(arr, dtype=float).flatten()
        sc = MinMaxScaler()
        processed[k] = sc.fit_transform(vec.reshape(-1, 1)).flatten()
        scalers[k] = sc

    n_weeks = feature_dict['load'].shape[0]
    need_weeks = encoder_len_weeks + decoder_len_weeks
    if n_weeks < need_weeks:
        raise ValueError(f"Need ≥{need_weeks} consecutive weeks, found {n_weeks}.")

    enc_seq_len = encoder_len_weeks * num_in_week
    dec_seq_len = decoder_len_weeks * num_in_week
    L = dec_seq_len - output_len
    if L <= 0:
        raise ValueError("`output_len` must be smaller than decoder sequence length.")

    ext_keys = [k for k in feature_dict.keys() if k != 'load']
    K_ext = len(ext_keys)

    X_enc_l, X_dec_in_l, Y_dec_target = [], [], []
    X_enc_ext, X_dec_in_ext = [], []
    last_start = n_weeks - need_weeks

    for w in range(last_start + 1):
        enc_start = w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start = enc_end
        dec_end   = dec_start + dec_seq_len

        enc_l = processed['load'][enc_start:enc_end]
        dec_full_l = processed['load'][dec_start:dec_end]

        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)
            dec_ext = np.stack([processed[k][dec_start:dec_start + L] for k in ext_keys], axis=-1)
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32)
            dec_ext = np.empty((L, 0), dtype=np.float32)

        targets = np.stack([dec_full_l[i:i+output_len] for i in range(L+1)], axis=0)

        X_enc_l.append(enc_l)
        X_dec_in_l.append(dec_full_l[:L])
        X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext)
        Y_dec_target.append(targets)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    data_tensors = {
        'X_enc_l'      : to_tensor(np.array(X_enc_l)).unsqueeze(-1),
        'X_enc_ext'    : to_tensor(np.array(X_enc_ext)),
        'X_dec_in_l'   : to_tensor(np.array(X_dec_in_l)).unsqueeze(-1),
        'X_dec_in_ext' : to_tensor(np.array(X_dec_in_ext)),
        'Y_dec_target' : to_tensor(np.array(Y_dec_target)).unsqueeze(-1),
    }

    for k, v in data_tensors.items():
        print(f"{k:15s} {tuple(v.shape)}")

    B = data_tensors['X_enc_l'].shape[0]
    split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}
    test_dict  = {k: v[split:] for k, v in data_tensors.items()}
    return train_dict, test_dict, scalers


# ====================== Seq2Seq processing (use existing scalers) ======================

def process_seq2seq_data_using_scalers(
    feature_dict,
    *,
    scalers,
    train_ratio=0.7,
    output_len=24,
    encoder_len_weeks=1,
    decoder_len_weeks=1,
    num_in_week=168,
    device=None,
):
    processed = {}
    for k, arr in feature_dict.items():
        if arr.size == 0:
            raise ValueError(f"feature '{k}' is empty.")
        vec = np.asarray(arr, dtype=float).flatten()
        if k not in scalers:
            raise KeyError(f"Missing scaler for key='{k}'. Keys={list(scalers.keys())}")
        sc = scalers[k]
        processed[k] = sc.transform(vec.reshape(-1, 1)).flatten()

    n_weeks = feature_dict['load'].shape[0]
    need_weeks = encoder_len_weeks + decoder_len_weeks
    if n_weeks < need_weeks:
        raise ValueError(f"Need ≥{need_weeks} consecutive weeks, found {n_weeks}.")

    enc_seq_len = encoder_len_weeks * num_in_week
    dec_seq_len = decoder_len_weeks * num_in_week
    L = dec_seq_len - output_len
    if L <= 0:
        raise ValueError("`output_len` must be smaller than decoder sequence length.")

    ext_keys = [k for k in feature_dict.keys() if k != 'load']
    K_ext = len(ext_keys)

    X_enc_l, X_dec_in_l, Y_dec_target = [], [], []
    X_enc_ext, X_dec_in_ext = [], []
    last_start = n_weeks - need_weeks

    for w in range(last_start + 1):
        enc_start = w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start = enc_end
        dec_end   = dec_start + dec_seq_len

        enc_l = processed['load'][enc_start:enc_end]
        dec_full_l = processed['load'][dec_start:dec_end]

        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)
            dec_ext = np.stack([processed[k][dec_start:dec_start + L] for k in ext_keys], axis=-1)
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32)
            dec_ext = np.empty((L, 0), dtype=np.float32)

        targets = np.stack([dec_full_l[i:i+output_len] for i in range(L+1)], axis=0)

        X_enc_l.append(enc_l)
        X_dec_in_l.append(dec_full_l[:L])
        X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext)
        Y_dec_target.append(targets)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    data_tensors = {
        'X_enc_l'      : to_tensor(np.array(X_enc_l)).unsqueeze(-1),
        'X_enc_ext'    : to_tensor(np.array(X_enc_ext)),
        'X_dec_in_l'   : to_tensor(np.array(X_dec_in_l)).unsqueeze(-1),
        'X_dec_in_ext' : to_tensor(np.array(X_dec_in_ext)),
        'Y_dec_target' : to_tensor(np.array(Y_dec_target)).unsqueeze(-1),
    }

    B = data_tensors['X_enc_l'].shape[0]
    split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}
    test_dict  = {k: v[split:] for k, v in data_tensors.items()}
    return train_dict, test_dict


# ===================== V5 Losses (Soft-threshold unified) =====================

def soft_threshold_mask(y_true: torch.Tensor, thr_frac: float, tau: float):
    B = y_true.size(0)
    y_flat = y_true.reshape(B, -1)
    y_max  = y_flat.max(dim=1, keepdim=True).values  # [B,1]
    thr = thr_frac * y_max
    thr = thr.view(B, 1, 1)  # broadcast
    w = torch.sigmoid((y_true - thr) / (tau + 1e-12))
    return w, thr


def gaussian_nll_pointwise(mu, logvar, y, logvar_min=-10.0, logvar_max=5.0):
    logvar = logvar.clamp(min=logvar_min, max=logvar_max)
    var = logvar.exp()
    log2pi = math.log(2.0 * math.pi)
    nll = 0.5 * (logvar + log2pi + (y - mu) ** 2 / (var + 1e-12))
    return nll, logvar


def soft_thr_region_mse(mu, y, w):
    err2 = (mu - y).pow(2)
    return (w * err2).sum() / (w.sum() + 1e-12)


def weighted_pinball(y, yq, q, w):
    pl = pinball_loss(y, yq, q)
    return (w * pl).sum() / (w.sum() + 1e-12)


def softargmax_time(y, temp: float):
    B, T = y.shape
    idx = torch.arange(T, device=y.device, dtype=y.dtype).view(1, T)
    p = torch.softmax(y / (temp + 1e-12), dim=1)
    t = (p * idx).sum(dim=1)
    return t


def v5_peak_fidelity_loss(
    mu_preds: torch.Tensor,
    logvar_preds: torch.Tensor,
    tgt: torch.Tensor,
    *,
    thr_frac: float = 0.90,
    tau: float = 0.03,
    q_upper: float = 0.95,
    softarg_temp: float = 0.05,
    lam_thr: float = 3.0,
    lam_q: float = 1.0,
    lam_time: float = 0.1,
    load_scale: float = 1.0,
    logvar_min: float = -10.0,
    logvar_max: float = 5.0,
    return_parts: bool = False,
):
    mu = mu_preds.squeeze(-1)
    y  = tgt.squeeze(-1)
    logv = logvar_preds.squeeze(-1)

    nll, logv_clamped = gaussian_nll_pointwise(mu, logv, y, logvar_min, logvar_max)
    sigma = (0.5 * logv_clamped).exp()

    w, _ = soft_threshold_mask(y, thr_frac=thr_frac, tau=tau)
    L_thr = soft_thr_region_mse(mu, y, w)

    zq = gaussian_icdf(q_upper, device=mu.device)
    yq = mu + zq * sigma
    L_q = weighted_pinball(y, yq, q_upper, w)

    # rescale to original load units
    L_thr = L_thr * (load_scale ** 2)
    L_q   = L_q   * load_scale

    B = y.size(0)
    y_flat  = y.reshape(B, -1)
    mu_flat = mu.reshape(B, -1)
    t_true = softargmax_time(y_flat, temp=softarg_temp)
    t_pred = softargmax_time(mu_flat, temp=softarg_temp)
    T = y_flat.size(1)
    L_time = (t_pred - t_true).abs().mean() / (T + 1e-12)

    nll_mean  = nll.mean()
    loss = nll_mean + lam_thr * L_thr + lam_q * L_q + lam_time * L_time

    if return_parts:
        parts = {
            "nll": float(nll_mean.detach().cpu()),
            "thr": float(L_thr.detach().cpu()),
            "q": float(L_q.detach().cpu()),
            "time": float(L_time.detach().cpu()),
            "lam_thr": float(lam_thr),
            "lam_q": float(lam_q),
            "lam_time": float(lam_time),
        }
        return loss, parts
    return loss


# ===================== V5 Train (BASE) =====================

def train_model_v5(
    model,
    train_loader,
    epochs,
    lr,
    device,
    top_k=1,
    kl_weight=0.001,
    warmup_epochs=10,
    save_path="best_model.pt",
    optimizer=None,
    # v5 params
    thr_frac: float = 0.90,
    tau: float = 0.03,
    q_upper: float = 0.95,
    softarg_temp: float = 0.05,
    lam_thr: float = 3.0,
    lam_q: float = 1.0,
    lam_time: float = 0.1,
    load_scale: float = 1.0,
    peak_warmup_epochs: int = 50,
    grad_clip_norm: float = 1.0,
    logvar_min: float = -10.0,
    logvar_max: float = 5.0,
):
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_train = float("inf")
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        skipped = 0
        sum_parts = {"nll":0.0, "thr":0.0, "q":0.0, "time":0.0, "kl":0.0}
        cnt_parts = 0

        w_peak = 1.0 if (peak_warmup_epochs is None or peak_warmup_epochs <= 0) else min(1.0, ep / float(peak_warmup_epochs))
        lam_thr_ep  = lam_thr  * w_peak
        lam_q_ep    = lam_q    * w_peak
        lam_time_ep = lam_time * w_peak

        for (enc_l, enc_ext, dec_l, dec_ext, tgt) in train_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = (
                enc_l.to(device),
                enc_ext.to(device),
                dec_l.to(device),
                dec_ext.to(device),
                tgt.to(device),
            )

            optimizer.zero_grad()

            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=top_k, warmup_epochs=warmup_epochs
            )

            loss_main, parts = v5_peak_fidelity_loss(
                mu_preds, logvar_preds, tgt,
                thr_frac=thr_frac,
                tau=tau,
                q_upper=q_upper,
                softarg_temp=softarg_temp,
                lam_thr=lam_thr_ep,
                lam_q=lam_q_ep,
                lam_time=lam_time_ep,
                load_scale=load_scale,
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                return_parts=True,
            )

            if not torch.isfinite(loss_main):
                skipped += 1
                continue

            if kl_weight is not None and kl_weight > 0:
                kl = kl_loss(mu_z, logvar_z)
            else:
                kl = torch.zeros((), device=device)

            loss = loss_main + kl_weight * kl
            if not torch.isfinite(loss):
                skipped += 1
                continue

            sum_parts["nll"]  += parts["nll"]
            sum_parts["thr"]  += parts["thr"]
            sum_parts["q"]    += parts["q"]
            sum_parts["time"] += parts["time"]
            sum_parts["kl"]   += float(kl.detach().cpu())
            cnt_parts += 1

            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            running += loss.item() * enc_l.size(0)

        denom = max(1, (len(train_loader.dataset) - skipped * train_loader.batch_size))
        avg = running / denom

        if avg < best_train:
            best_train = avg
            best_epoch = ep
            torch.save(model.state_dict(), save_path)
            print(f"✅ [BASE v5_v1temp] Saved best model at epoch {ep} | loss {best_train:.6f}")

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(
                f"[BASE v5_v1temp] Epoch {ep:3d}/{epochs} | "
                f"train loss: {avg:.6f} | best: {best_train:.6f} (ep {best_epoch}) | "
                f"lam_thr={lam_thr_ep:.3f} lam_q={lam_q_ep:.3f} lam_time={lam_time_ep:.3f} | "
                f"skipped={skipped}"
            )
            if cnt_parts > 0:
                avg_nll  = sum_parts["nll"]  / cnt_parts
                avg_thr  = sum_parts["thr"]  / cnt_parts
                avg_q    = sum_parts["q"]    / cnt_parts
                avg_time = sum_parts["time"] / cnt_parts
                avg_kl   = sum_parts["kl"]   / cnt_parts
                print(
                    f"[PARTS] nll={avg_nll:.4f} thr={avg_thr:.4f} q={avg_q:.4f} "
                    f"time={avg_time:.4f} kl={avg_kl:.4f} | weighted: "
                    f"+{lam_thr_ep*avg_thr:.4f} +{lam_q_ep*avg_q:.4f} +{lam_time_ep*avg_time:.4f} +{kl_weight*avg_kl:.4f}"
                )

    print(f"\n🏁 [BASE v5_v1temp] Done. Best epoch {best_epoch} | loss {best_train:.6f}")
    return model


# ===================== Peak eval helper (unchanged) =====================

@torch.no_grad()
def _evaluate_peak_batch(preds_4d, tgts_4d):
    B, Lp1, out_len, _ = preds_4d.shape
    reports = []

    for b in range(B):
        pred_week = reconstruct_sequence(preds_4d[b, :, :, 0])
        true_week = reconstruct_sequence(tgts_4d[b, :, :, 0])

        T = min(int(pred_week.shape[0]), int(true_week.shape[0]))
        pred_week = pred_week[:T]
        true_week = true_week[:T]

        points_per_day = max(T // 7, 1)
        dt_min = int(round(1440 / points_per_day))
        cfg = ercot_defaults(dt_min=dt_min)

        rep = peak_metrics_week(
            y_true_week=true_week.detach().cpu().numpy() if torch.is_tensor(true_week) else np.asarray(true_week),
            y_pred_week=pred_week.detach().cpu().numpy() if torch.is_tensor(pred_week) else np.asarray(pred_week),
            points_per_day=points_per_day,
            **cfg,
        )
        reports.append(rep)

    def _mean_over_batch(key, where="summary"):
        vals = []
        for r in reports:
            v = r.get(where, {}).get(key, float("nan"))
            if not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
        return float(np.mean(vals)) if len(vals) else float("nan")

    pmse_union_mean = _mean_over_batch("ThrPeakMSE_union_mean")
    log = {
        "PVPE_mean": _mean_over_batch("PVPE_mean"),
        "PTE_min_mean": _mean_over_batch("PTE_min_mean"),
        "PeakWindowIoU_mean": _mean_over_batch("PeakWindowIoU_mean"),
        "ThrWindowIoU_mean": _mean_over_batch("ThrWindowIoU_mean"),
        "PeakPeriodMSE_thr_mean": pmse_union_mean,
    }
    return log, reports


# ===================== Evaluation (export xlsx) =====================

@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    loss_fn,
    device,
    data_name=None,
    model_name=None,
    quantiles=(0.1, 0.5, 0.9),
    alpha=0.1,
    data_export_list=None,
):
    print(f"--- Evaluating model '{model_name}' on data '{data_name}' ---")
    model.to(device)
    model.eval()

    running_mse = 0.0
    running_nll = 0.0
    running_crps = 0.0
    running_qpin = 0.0
    running_wink = 0.0

    peak_sum = {
        "PVPE_mean": 0.0,
        "PTE_min_mean": 0.0,
        "PeakWindowIoU_mean": 0.0,
        "ThrWindowIoU_mean": 0.0,
        "PeakPeriodMSE_thr_mean": 0.0,
    }
    peak_count = 0

    for batch_id, batch in enumerate(test_loader):
        enc_l, enc_ext, dec_l, dec_ext, tgt = [t.to(device) for t in batch]
        B = enc_l.size(0)

        mu_preds, logvar_preds, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
        mu4d  = mu_preds
        tgt4d = tgt

        mu_preds = mu_preds.squeeze(-1)
        logvar_preds = logvar_preds.squeeze(-1)
        tgt = tgt.squeeze(-1)

        peak_log, _ = _evaluate_peak_batch(mu4d, tgt4d)
        for k in peak_sum:
            v = peak_log.get(k, float("nan"))
            if not (isinstance(v, float) and np.isnan(v)):
                peak_sum[k] += float(v)
        peak_count += 1

        # same as your prior: evaluate offset=0, horizon=Lp1
        mu_first = mu_preds[:, :, 0]
        logvar_first = logvar_preds[:, :, 0]
        tgt_first = tgt[:, :, 0]
        sigma_first = logvar_first.exp().sqrt()

        running_mse += loss_fn(mu_first, tgt_first).item() * B

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
            ql = pinball_loss(tgt_first, yq, q).mean()
            q_losses.append(ql)
        qpin_mean = torch.stack(q_losses).mean()
        running_qpin += qpin_mean.item() * B

        z = gaussian_icdf(1.0 - alpha/2.0, device=mu_first.device)
        Lb = mu_first - z * sigma_first
        Ub = mu_first + z * sigma_first
        ws = winkler_score(tgt_first, Lb, Ub, alpha).mean()
        running_wink += ws.item() * B

        # export
        if data_export_list is not None and model_name is not None:
            for i in range(B):
                hist_i = enc_l[i].detach().cpu().squeeze(-1)
                mu_i = mu_first[i].detach().cpu()
                y_true_i = tgt_first[i].detach().cpu()
                std_i = sigma_first[i].detach().cpu()

                Lh = len(hist_i)
                H = len(mu_i)
                x_hist = np.arange(Lh)
                x_fore = np.arange(Lh, Lh + H)

                df_hist = pd.DataFrame({'time_step': x_hist, 'value': hist_i.numpy(), 'value_type': 'history'})
                df_true = pd.DataFrame({'time_step': x_fore, 'value': y_true_i.numpy(), 'value_type': 'true'})
                df_pred = pd.DataFrame({'time_step': x_fore, 'value': mu_i.numpy(), 'value_type': 'pred_mean'})
                df_std  = pd.DataFrame({'time_step': x_fore, 'value': std_i.numpy(), 'value_type': 'pred_std'})
                df_sample = pd.concat([df_hist, df_true, df_pred, df_std], ignore_index=True)
                df_sample['model_name'] = model_name
                df_sample['sample_index'] = i + batch_id * test_loader.batch_size
                data_export_list.append(df_sample)

    test_mse = running_mse / len(test_loader.dataset)
    horizon = mu_first.size(1)
    test_nll  = running_nll  / (len(test_loader.dataset) * horizon)
    test_crps = running_crps / len(test_loader.dataset)
    test_qpin = running_qpin / len(test_loader.dataset)
    test_wink = running_wink / len(test_loader.dataset)

    print(f"\n🧪 Test MSE         : {test_mse:.6f}")
    print(f"🧪 Test CRPS        : {test_crps:.6f}")
    print(f"🧪 Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
    print(f"🧪 Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
    if peak_count > 0:
        for k in peak_sum:
            peak_sum[k] /= peak_count
        pmse_thr = peak_sum.get("PeakPeriodMSE_thr_mean", float("nan"))
        print(
            f"[PEAK] PVPE_mean={peak_sum['PVPE_mean']:.4f}  "
            f"PTE_min_mean={peak_sum['PTE_min_mean']:.1f} min  "
            f"IoU(maxwin)={peak_sum['PeakWindowIoU_mean']:.3f}  "
            f"IoU(thr)={peak_sum['ThrWindowIoU_mean']:.3f}  "
            f"PMSE_thr={pmse_thr:.6f}"
        )
    print("---------------------------------------------------\n")

    return test_mse, test_nll, test_crps, test_qpin, test_wink


# ===================== MAIN (BASE ONLY, v1_temp) =====================

if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    # ------------------ switches ------------------
    EVAL_XFMR = "data1"

    batch_size   = 16
    epochs_base  = 1000
    lr           = 1e-3
    KL_BASE_WEIGHT = 0.001

    # V5 hyperparams
    V5_THR_FRAC   = 0.90
    V5_TAU        = 0.10
    V5_Q_UPPER    = 0.90
    V5_SOFTARG_T  = 0.15
    V5_LAM_THR    = 0.1
    V5_LAM_Q      = 0.05
    V5_LAM_TIME   = 0.02
    V5_PEAK_WARM  = 300

    xprime_dim  = 40
    hidden_dim  = 64
    latent_dim  = 32
    num_layers  = 4
    output_len  = 24
    encoder_len_weeks = 1
    input_dim   = 1
    output_dim  = 1

    top_k      = 1
    warmup_ep  = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Naming (v1_temp) ----------
    TAG = "v5_v1temp_oracle"
    data_name  = "Oncor_load"
    model_name = f"M2OE2_v1_temp_{output_len}h_Enc{encoder_len_weeks}w"

    BASE_CKPT = f"{data_name}_{model_name}_{TAG}_base_best.pt"
    SCALER_META_JSON = f"vae_base_scaler_meta_{TAG}.json"
    TRAIN_CFG_JSON   = f"train_config_{TAG}.json"
    XLSX_EXPORT_PATH = f"vae_base_only_{TAG}_XFMR{EVAL_XFMR}.xlsx"

    print(f"[INFO] BASE_CKPT: {BASE_CKPT}")
    print(f"[INFO] SCALER_META_JSON: {SCALER_META_JSON}")
    print(f"[INFO] TRAIN_CFG_JSON: {TRAIN_CFG_JSON}")
    print(f"[INFO] XLSX_EXPORT_PATH: {XLSX_EXPORT_PATH}")

    # ============================================================
    # 0) Load ALL data (all XFMR), build oracle temp_fc_* features, fit scalers ONCE
    # ============================================================
    times, load, temp, workday, season = get_data_oncor_load_weekly(XFMR="all")
    print("[TRAIN] load shape (all XFMR):", load.shape)
    print("[TRAIN] temp shape (all XFMR):", temp.shape)

    # oracle forecast summaries from TRUE temp
    temp_flat = np.asarray(temp, dtype=float).flatten()
    mean24, max24, min24, ramp24 = compute_forward_24h_temp_stats(temp_flat, horizon=24)

    # reshape back to temp shape
    mean24 = mean24.reshape(temp.shape)
    max24  = max24.reshape(temp.shape)
    min24  = min24.reshape(temp.shape)
    ramp24 = ramp24.reshape(temp.shape)

    feature_dict_all = {
        "load": load,
        "temp": temp,
        "workday": workday,
        "season": season,
        # >>> v1_temp oracle features <<<
        "temp_fc_mean24": mean24,
        "temp_fc_max24":  max24,
        "temp_fc_min24":  min24,
        "temp_fc_ramp24": ramp24,
    }

    train_data_all, test_data_all, scalers = process_seq2seq_data(
        feature_dict      = feature_dict_all,
        train_ratio       = 0.7,
        output_len        = output_len,
        encoder_len_weeks = encoder_len_weeks,
        device            = device,
    )

    # Save scaler meta (include new keys too)
    scaler_meta = {}
    for k, sc in scalers.items():
        scaler_meta[f"{k}_min"] = float(sc.data_min_[0])
        scaler_meta[f"{k}_max"] = float(sc.data_max_[0])
    with open(SCALER_META_JSON, "w", encoding="utf-8") as f:
        json.dump(scaler_meta, f, indent=2)
    print(f"[INFO] Saved scaler meta to {SCALER_META_JSON}")

    sc_load = scalers["load"]
    load_scale = float(sc_load.data_max_[0] - sc_load.data_min_[0])
    print(f"[INFO] load_scale (kW) = {load_scale:.6f}")

    n_externals = train_data_all["X_enc_ext"].shape[-1]
    print(f"[INFO] K_ext (number of external features) = {n_externals}")

    train_loader_all = make_loader(train_data_all, batch_size=batch_size, shuffle=True)

    # Save training config for reproducibility
    train_cfg = dict(
        TAG=TAG,
        seed=seed,
        data_name=data_name,
        model_name=model_name,
        EVAL_XFMR=EVAL_XFMR,
        batch_size=batch_size,
        epochs_base=epochs_base,
        lr=lr,
        KL_BASE_WEIGHT=KL_BASE_WEIGHT,
        V5=dict(
            thr_frac=V5_THR_FRAC,
            tau=V5_TAU,
            q_upper=V5_Q_UPPER,
            softarg_temp=V5_SOFTARG_T,
            lam_thr=V5_LAM_THR,
            lam_q=V5_LAM_Q,
            lam_time=V5_LAM_TIME,
            peak_warmup_epochs=V5_PEAK_WARM,
        ),
        model_dims=dict(
            xprime_dim=xprime_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            output_len=output_len,
            encoder_len_weeks=encoder_len_weeks,
            input_dim=input_dim,
            output_dim=output_dim,
            n_externals=n_externals,
            top_k=top_k,
            warmup_ep=warmup_ep,
        ),
        oracle_temp_fc_features=True,
        temp_fc_horizon=24,
        feature_keys=list(feature_dict_all.keys()),
    )
    with open(TRAIN_CFG_JSON, "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=2)
    print(f"[INFO] Saved training config to {TRAIN_CFG_JSON}")

    # ============================================================
    # 1) Build XFMR-specific dataset using SAME scalers (NO refit)
    #    - eval/plot: use ALL samples as test (train_ratio=0.0)
    # ============================================================
    print(f"[XFMR] Building eval dataset for XFMR={EVAL_XFMR} using BASE scalers...")

    times_e, load_e, temp_e, workday_e, season_e = get_data_oncor_load_weekly(XFMR=EVAL_XFMR)

    temp_e_flat = np.asarray(temp_e, dtype=float).flatten()
    mean24_e, max24_e, min24_e, ramp24_e = compute_forward_24h_temp_stats(temp_e_flat, horizon=24)

    mean24_e = mean24_e.reshape(temp_e.shape)
    max24_e  = max24_e.reshape(temp_e.shape)
    min24_e  = min24_e.reshape(temp_e.shape)
    ramp24_e = ramp24_e.reshape(temp_e.shape)

    feature_dict_xfmr = {
        "load": load_e,
        "temp": temp_e,
        "workday": workday_e,
        "season": season_e,
        "temp_fc_mean24": mean24_e,
        "temp_fc_max24":  max24_e,
        "temp_fc_min24":  min24_e,
        "temp_fc_ramp24": ramp24_e,
    }

    _, test_data_eval_all = process_seq2seq_data_using_scalers(
        feature_dict      = feature_dict_xfmr,
        scalers           = scalers,
        train_ratio       = 0.0,
        output_len        = output_len,
        encoder_len_weeks = encoder_len_weeks,
        device            = device,
    )
    test_loader_eval = make_loader(test_data_eval_all, batch_size=batch_size, shuffle=False)

    plot_data_frames = []

    # ============================================================
    # 2) Train BASE model (v5_v1temp_oracle) if needed
    # ============================================================
    if not os.path.exists(BASE_CKPT):
        print(f"\n[WARN] Base ckpt not found: {BASE_CKPT}")
        print("[INFO] Training BASE model (V5 + v1_temp oracle features) from scratch...")

        model_base_train = VariationalSeq2Seq_meta(
            xprime_dim  = xprime_dim,
            input_dim   = input_dim,
            hidden_size = hidden_dim,
            latent_size = latent_dim,
            output_len  = output_len,
            n_externals = n_externals,
            output_dim  = output_dim,
            num_layers  = num_layers,
            dropout     = 0.1,
        ).to(device)

        optimizer_base = AdamW(model_base_train.parameters(), lr=lr, weight_decay=1e-4)

        train_model_v5(
            model        = model_base_train,
            train_loader = train_loader_all,
            epochs       = epochs_base,
            lr           = lr,
            device       = device,
            top_k        = top_k,
            kl_weight    = KL_BASE_WEIGHT,
            warmup_epochs= warmup_ep,
            save_path    = BASE_CKPT,
            optimizer    = optimizer_base,
            thr_frac     = V5_THR_FRAC,
            tau          = V5_TAU,
            q_upper      = V5_Q_UPPER,
            softarg_temp = V5_SOFTARG_T,
            lam_thr      = V5_LAM_THR,
            lam_q        = V5_LAM_Q,
            lam_time     = V5_LAM_TIME,
            peak_warmup_epochs = V5_PEAK_WARM,
            grad_clip_norm     = 1.0,
            logvar_min         = -10.0,
            logvar_max         = 5.0,
            load_scale         = load_scale,
        )
        print(f"[✓] BASE model trained & saved to {BASE_CKPT}\n")
    else:
        print(f"[✓] Found existing BASE ckpt: {BASE_CKPT}\n")

    # ============================================================
    # 3) Eval BASE on XFMR + export XLSX
    # ============================================================
    print(f"[EVAL] Evaluating BASE model on XFMR={EVAL_XFMR} ...")

    model_base = VariationalSeq2Seq_meta(
        xprime_dim  = xprime_dim,
        input_dim   = input_dim,
        hidden_size = hidden_dim,
        latent_size = latent_dim,
        output_len  = output_len,
        n_externals = n_externals,
        output_dim  = output_dim,
        num_layers  = num_layers,
        dropout     = 0.1,
    ).to(device)

    state_base = torch.load(BASE_CKPT, map_location=device)
    model_base.load_state_dict(state_base, strict=True)
    print(f"[✓] BASE weights loaded from {BASE_CKPT}")

    evaluate_model(
        model           = model_base,
        test_loader     = test_loader_eval,
        loss_fn         = nn.MSELoss(),
        device          = device,
        data_name       = f"{data_name}_xfmr{EVAL_XFMR}_{TAG}",
        model_name      = f"VAE_BASE_{TAG}",  # stable plotting key
        data_export_list= plot_data_frames,
    )

    if plot_data_frames:
        final_plot_df = pd.concat(plot_data_frames, ignore_index=True)
        final_plot_df["XFMR"] = int(EVAL_XFMR)
        final_plot_df.to_excel(XLSX_EXPORT_PATH, index=False)
        print(f"[✓] Plot data saved to {XLSX_EXPORT_PATH}")
    else:
        print("[WARN] No plot_data_frames collected; XLSX not saved.")
