import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler
from data_utils import *
from model_v1 import *
import numpy as np, random, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import math
from data_utils import reconstruct_sequence
from peak_metrics import peak_metrics_week, ercot_defaults
from lora_utils import LoRALinear, freeze_all, collect_lora_params, add_lora_to_parameter, DeltaLinear, collect_adapter_params
from torch.optim import AdamW

# ------------------------------------------------------------------------------
# Reproducibility and Helper Functions
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across CPU and GPU operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gaussian_icdf(p, device):
    """
    Computes the inverse cumulative distribution function (probit) for a Gaussian.
    """
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        2 * torch.as_tensor(p, device=device) - 1
    )

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) for two boolean masks.
    """
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union + 1e-12)

def _compute_week_peak_metrics(
    y_true_week: np.ndarray,
    y_pred_week: np.ndarray,
    *,
    peak_window_min: int = 240,
    thr_frac: float = 0.90
):
    """
    Computes peak-specific metrics (PVPE, PTE, IoU) for a weekly sequence.
    """
    T = int(len(y_true_week)); assert T == int(len(y_pred_week)), "Prediction and Ground Truth must have equal length."
    points_per_day = T // 7 if T >= 7 else T; points_per_day = max(1, points_per_day)
    dt_min = int(round(1440 / points_per_day)); half_w = int(round(0.5 * peak_window_min / dt_min))
    pvpe_list, pte_list, iou_fix_list, iou_thr_list = [], [], [], []
    for d in range(7):
        s, e = d * points_per_day, (d + 1) * points_per_day
        yt = y_true_week[s:e]; yp = y_pred_week[s:e]
        if len(yt) == 0 or len(yp) == 0: continue
        it = int(np.argmax(yt)); vt = float(yt[it]); ip = int(np.argmax(yp)); vp = float(yp[ip])
        pvpe_list.append(abs(vp - vt)); pte_list.append(abs(ip - it) * dt_min)
        mt = np.zeros_like(yt, dtype=bool); mp = np.zeros_like(yp, dtype=bool)
        mt[max(0, it - half_w):min(points_per_day, it + half_w + 1)] = True
        mp[max(0, ip - half_w):min(points_per_day, ip + half_w + 1)] = True
        iou_fix_list.append(_iou(mt, mp))
        mt2 = yt >= (thr_frac * vt); mp2 = yp >= (thr_frac * vp)
        iou_thr_list.append(_iou(mt2, mp2))
    def _safe_mean(arr): return float(np.mean(arr)) if len(arr) else float("nan")
    out = {"PVPE_mean": _safe_mean(pvpe_list), "PTE_min_mean": _safe_mean(pte_list),
           "PeakWindowIoU_mean": _safe_mean(iou_fix_list), "ThrWindowIoU_mean": _safe_mean(iou_thr_list),}
    return out

def pinball_loss(y, yq, q):
    e = y - yq; return torch.where(e >= 0, q * e, (q - 1) * e)

def winkler_score(y, L, U, alpha):
    width = (U - L); below = (L - y).clamp(min=0.0); above = (y - U).clamp(min=0.0)
    return width + (2.0 / alpha) * (below + above)

def crps_gaussian(mu, logvar, target):
    std = (0.5 * logvar).exp(); z = (target - mu) / std
    normal = Normal(torch.zeros_like(z), torch.ones_like(z)); phi = torch.exp(normal.log_prob(z)); Phi = normal.cdf(z)
    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi)); return crps.mean()

def make_loader(split_dict, batch_size, shuffle):
    ds = TensorDataset(split_dict['X_enc_l'], split_dict['X_enc_ext'], split_dict['X_dec_in_l'], split_dict['X_dec_in_ext'], split_dict['Y_dec_target'],)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def process_seq2seq_data(
        feature_dict, *, train_ratio=0.7, norm_features=('load', 'temp'), output_len=24,
        encoder_len_weeks=1, decoder_len_weeks=1, num_in_week=168, device=None):
    """
    Data preprocessing pipeline: normalization, sequence construction, and tensor packing.
    """
    processed, scalers = {}, {}
    for k, arr in feature_dict.items():
        if arr.size == 0: raise ValueError(f"feature '{k}' is empty.")
        vec = np.asarray(arr, dtype=float).flatten(); sc = MinMaxScaler()
        processed[k] = sc.fit_transform(vec.reshape(-1, 1)).flatten(); scalers[k] = sc
    n_weeks = feature_dict['load'].shape[0]; need_weeks = encoder_len_weeks + decoder_len_weeks
    if n_weeks < need_weeks: raise ValueError(f"Need ≥{need_weeks} consecutive weeks, found {n_weeks}.")
    enc_seq_len = encoder_len_weeks * num_in_week; dec_seq_len = decoder_len_weeks * num_in_week
    L = dec_seq_len - output_len;
    if L <= 0: raise ValueError("`output_len` must be smaller than decoder sequence length.")
    ext_keys = [k for k in feature_dict.keys() if k != 'load']; K_ext = len(ext_keys)
    X_enc_l, X_dec_in_l, Y_dec_target = [], [], []; X_enc_ext, X_dec_in_ext = [], []
    last_start = n_weeks - need_weeks
    for w in range(last_start + 1):
        enc_start =  w * num_in_week; enc_end = (w + encoder_len_weeks) * num_in_week
        dec_start =  enc_end; dec_end =  dec_start + dec_seq_len
        enc_l = processed['load'][enc_start:enc_end]; dec_full_l = processed['load'][dec_start:dec_end]
        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)
            dec_ext = np.stack([processed[k][dec_start: dec_start + L] for k in ext_keys], axis=-1)
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32); dec_ext = np.empty((L, 0), dtype=np.float32)
        targets = np.stack([dec_full_l[i:i+output_len] for i in range(L+1)], axis=0)
        X_enc_l.append(enc_l); X_dec_in_l.append(dec_full_l[:L]); X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext); Y_dec_target.append(targets)
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    data_tensors = {
        'X_enc_l'      : to_tensor(np.array(X_enc_l)).unsqueeze(-1), 'X_enc_ext'    : to_tensor(np.array(X_enc_ext)),
        'X_dec_in_l'   : to_tensor(np.array(X_dec_in_l)).unsqueeze(-1), 'X_dec_in_ext' : to_tensor(np.array(X_dec_in_ext)),
        'Y_dec_target' : to_tensor(np.array(Y_dec_target)).unsqueeze(-1),
    }
    for k, v in data_tensors.items(): print(f"{k:15s} {tuple(v.shape)}")
    B = data_tensors['X_enc_l'].shape[0]; split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}; test_dict  = {k: v[split:] for k, v in data_tensors.items()}
    return train_dict, test_dict, scalers


# ------------------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------------------
def gaussian_nll_loss(mu, logvar, target, reduction='mean'):
    """
    Gaussian Negative Log-Likelihood (NLL) Loss.
    
    Updated to support a 'reduction' argument ('mean', 'sum', or 'none'),
    enabling element-wise weighting for peak-aware optimization.
    """
    # Clamp log-variance for numerical stability
    logvar = torch.clamp(logvar, min=-10, max=10)
    
    # Compute variance
    variance = torch.exp(logvar)
    
    # NLL components
    loss_term_1 = 0.5 * logvar
    loss_term_2 = 0.5 * torch.log(torch.tensor(2 * np.pi, device=mu.device))
    loss_term_3 = 0.5 * ((target - mu) ** 2) / variance
    
    nll = loss_term_1 + loss_term_2 + loss_term_3
    
    # Apply reduction
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    elif reduction == 'none':
        return nll # Return element-wise loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------

def train_model(model, train_loader, epochs, lr, device, 
                top_k=2, kl_weight=0.01, warmup_epochs=10, 
                save_path="best_model.pt", optimizer=None,
                # --- Peak Weighting Parameters ---
                peak_loss_weight: float = 10.0,
                peak_threshold_q: float = 0.90
                # ---------------------------------
               ):
    """
    Model training loop with Peak-Weighted Loss.
    
    This function implements a custom loss mechanism that assigns higher weights 
    to data points exceeding a specified quantile threshold, improving performance 
    on extreme events.
    """
    
    if optimizer is None:
        print("[Train] No optimizer provided, creating default AdamW for all params.")
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        print("[Train] Using provided optimizer (likely for LoRA).")
        pass 

    best_train = float("inf")
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for (enc_l, enc_ext, dec_l, dec_ext, tgt) in train_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = enc_l.to(device), enc_ext.to(device), dec_l.to(device), dec_ext.to(device), tgt.to(device)
            optimizer.zero_grad()

            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=top_k, warmup_epochs=warmup_epochs
            )
            
            # --- Peak-Weighted Loss Logic ---

            # 1. Calculate base NLL loss for all data points (no reduction)
            nll_all_points = gaussian_nll_loss(mu_preds, logvar_preds, tgt, reduction="none")

            # 2. Identify peak events based on the specified quantile threshold
            with torch.no_grad():
                # Determine the threshold value for this batch
                peak_thresh_val = torch.quantile(tgt.data, peak_threshold_q)
                # Create boolean mask for peak events
                is_peak = (tgt >= peak_thresh_val)

            # 3. Construct the weight tensor
            # Base weight is 1.0; peak points receive the specified higher weight
            weights = torch.ones_like(tgt, device=device)
            weights[is_peak] = peak_loss_weight

            # 4. Compute weighted NLL loss
            nll_weighted = (nll_all_points * weights).mean()
            
            # 5. Compute KL divergence loss
            kl = kl_loss(mu_z, logvar_z)
            
            # 6. Aggregate final loss
            loss = nll_weighted + kl_weight * kl
            
            # --- End of Loss Logic ---

            loss.backward()
            optimizer.step()

            running += loss.item() * enc_l.size(0)

        avg = running / len(train_loader.dataset)
        
        # Save model only on improvement
        if avg < best_train:
            best_train = avg
            best_epoch = ep
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {ep} | loss {best_train:.6f}")

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | train loss: {avg:.6f} | best: {best_train:.6f} (ep {best_epoch})")

    print(f"\n🏁 Done. Best epoch {best_epoch} | loss {best_train:.6f}")
    return model


@torch.no_grad()
def _evaluate_peak_batch(preds_4d, tgts_4d):
    """
    Helper function to batch-process peak metrics.
    """
    B, Lp1, out_len, _ = preds_4d.shape; reports = []
    for b in range(B):
        pred_week = reconstruct_sequence(preds_4d[b, :, :, 0]); true_week = reconstruct_sequence(tgts_4d [b, :, :, 0])
        T = min(int(pred_week.shape[0]), int(true_week.shape[0])); pred_week = pred_week[:T]; true_week = true_week[:T]
        points_per_day = max(T // 7, 1); dt_min = int(round(1440 / points_per_day)); cfg = ercot_defaults(dt_min=dt_min)
        rep = peak_metrics_week(
            y_true_week=true_week.detach().cpu().numpy() if torch.is_tensor(true_week) else np.asarray(true_week),
            y_pred_week=pred_week.detach().cpu().numpy() if torch.is_tensor(pred_week) else np.asarray(pred_week),
            points_per_day=points_per_day, **cfg)
        reports.append(rep)
    def _mean_over_batch(key, where="summary"):
        vals = [];
        for r in reports:
            v = r.get(where, {}).get(key, float("nan"))
            if not (isinstance(v, float) and np.isnan(v)): vals.append(v)
        return float(np.mean(vals)) if len(vals) else float("nan")
    pmse_union_mean = _mean_over_batch("ThrPeakMSE_union_mean")
    log = {"PVPE_mean": _mean_over_batch("PVPE_mean"), "PTE_min_mean": _mean_over_batch("PTE_min_mean"),
           "PeakWindowIoU_mean": _mean_over_batch("PeakWindowIoU_mean"), "ThrWindowIoU_mean": _mean_over_batch("ThrWindowIoU_mean"),
           "PeakPeriodMSE_thr_mean": pmse_union_mean,}
    return log, reports


@torch.no_grad()
def evaluate_model(model, test_loader, loss_fn, device,
                   model_path="model.pt", reduce="first", visualize=True,
                   quantiles=(0.1, 0.5, 0.9), alpha=0.1,
                   data_name=None, model_name=None, data_export_list=None):
    
    print(f"--- Evaluating model '{model_name}' on data '{data_name}' ---")
    
    print(f"Evaluating model '{model_name}' from its current in-memory state.")
    # Note: model state loading is handled externally to allow flexible evaluation of in-memory models.
    # model.load_state_dict(torch.load(model_path, map_location=device), strict=False) 

    model.to(device)
    model.eval()

    running_mse = 0.0; running_nll = 0.0; running_crps = 0.0; running_qpin = 0.0; running_wink = 0.0
    peak_sum = {"PVPE_mean": 0.0, "PTE_min_mean": 0.0, "PeakWindowIoU_mean": 0.0,
                "ThrWindowIoU_mean": 0.0, "PeakPeriodMSE_thr_mean": 0.0,}
    peak_count = 0; reports = []; running_pve_abs = 0.0; running_pve_pct = 0.0; all_preds = []; all_targets = []
    
    for batch in test_loader:
        if len(batch) == 5:
            enc_l, enc_ext, dec_l, dec_ext, tgt = [t.to(device) for t in batch]
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} tensors.")

        B = enc_l.size(0)
        mu_preds, logvar_preds, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
        mu4d  = mu_preds; tgt4d = tgt
        mu_preds = mu_preds.squeeze(-1); logvar_preds = logvar_preds.squeeze(-1); tgt = tgt.squeeze(-1)

        peak_log, _ = _evaluate_peak_batch(mu4d, tgt4d)
        for k in peak_sum:
            v = peak_log.get(k, float("nan"))
            if not (isinstance(v, float) and np.isnan(v)): peak_sum[k] += float(v)
        peak_count += 1

        if reduce == "first":
            mu_first = mu_preds[:, :, 0]; logvar_first = logvar_preds[:, :, 0]
            tgt_first = tgt[:, :, 0]; sigma_first = logvar_first.exp().sqrt()

            all_preds.extend(mu_first.cpu()); all_targets.extend(tgt_first.cpu())
            running_mse += loss_fn(mu_first, tgt_first).item() * B
            
            p_true, _ = tgt_first.max(dim=1); p_pred, _ = mu_first.max(dim=1)
            diff = (p_pred - p_true).abs()
            running_pve_abs += diff.sum().item(); running_pve_pct += (diff / (p_true + 1e-12)).sum().item()
            
            # Use the updated element-wise NLL for consistency (though reduction is manual here)
            nll_unreduced = gaussian_nll_loss(mu_first, logvar_first, tgt_first, reduction='none')
            running_nll += nll_unreduced.sum().item()

            crps = crps_gaussian(mu_first, logvar_first, tgt_first); running_crps += crps.item() * B
            q_losses = []
            for q in quantiles:
                zq = gaussian_icdf(q, device=mu_first.device); yq = mu_first + sigma_first * zq
                ql = pinball_loss(tgt_first, yq, q).mean(); q_losses.append(ql)
            qpin_mean = torch.stack(q_losses).mean(); running_qpin += qpin_mean.item() * B
            z = gaussian_icdf(1.0 - alpha/2.0, device=mu_first.device)
            Lb = mu_first - z * sigma_first; Ub = mu_first + z * sigma_first
            ws = winkler_score(tgt_first, Lb, Ub, alpha).mean(); running_wink += ws.item() * B

            def _fname(tag, i=None):
                mn = model_name if model_name is not None else "model"
                if i is None: return f"./result/{mn}_{tag}.pdf"
                return f"./result/{mn}_{tag}_{i}.pdf"

            if visualize:
                # (1) pred_only plots
                for i in range(min(5, mu_first.size(0))):
                    std_pred = sigma_first[i].cpu()
                    plt.figure(figsize=(4, 2)); plt.plot(tgt_first[i].cpu(), label='True', linestyle='--', color='red')
                    plt.plot(mu_first[i].cpu(), label='Mean Predicted', alpha=0.6, color='blue')
                    plt.fill_between(np.arange(mu_first.size(1)), mu_first[i].cpu() - std_pred, mu_first[i].cpu() + std_pred, color='blue', alpha=0.1, label='±1 Std Predicted')
                    plt.ylim(0, 1); plt.yticks([0, 0.5, 1], fontsize=14); plt.xticks(fontsize=14); plt.tight_layout()
                    plt.savefig(_fname("pred_only", i)); plt.close()
                    
                # (2) individual plots with historical data
                for i in range(min(5, mu_first.size(0))):
                    std_pred = sigma_first[i].cpu(); mu_i = mu_first[i].cpu(); y_true_i = tgt_first[i].cpu()
                    hist_i = enc_l[i].cpu().squeeze(-1); Lh = len(hist_i); H  = len(mu_i)
                    x_hist = np.arange(Lh); x_fore = np.arange(Lh, Lh + H)
                    plt.figure(figsize=(10, 2.5)); plt.plot(x_hist, hist_i, color='black', linewidth=1.5, label='History')
                    plt.plot(x_fore, y_true_i, '--', color='red', linewidth=1.5, label='True')
                    plt.plot(x_fore, mu_i, color='blue', alpha=0.8, linewidth=1.5, label='Mean Pred')
                    plt.fill_between(x_fore, mu_i - std_pred, mu_i + std_pred, color='blue', alpha=0.1, label='±1 $\sigma$ (pred.)')
                    plt.axvline(Lh - 1, color='grey', linestyle='--', alpha=0.6)
                    plt.xlim(0, Lh + H); plt.ylim(0, 1); plt.tight_layout(); plt.yticks([0, 0.5, 1], fontsize=14); plt.xticks(fontsize=14); plt.legend()

                    # --- Data Export ---
                    if data_export_list is not None and model_name is not None:
                        df_hist = pd.DataFrame({'time_step': x_hist, 'value': hist_i.numpy(), 'value_type': 'history'})
                        df_true = pd.DataFrame({'time_step': x_fore, 'value': y_true_i.numpy(), 'value_type': 'true'})
                        df_pred = pd.DataFrame({'time_step': x_fore, 'value': mu_i.numpy(), 'value_type': 'pred_mean'})
                        df_std = pd.DataFrame({'time_step': x_fore, 'value': std_pred.numpy(), 'value_type': 'pred_std'}) 
                        df_sample = pd.concat([df_hist, df_true, df_pred, df_std])
                        df_sample['model_name'] = model_name
                        df_sample['sample_index'] = i
                        data_export_list.append(df_sample)
                        print(f"[Data Export] Saved sample {i} for model {model_name}")
                    # --- End Data Export ---

                    plt.savefig(_fname("with_hist", i)); plt.close()
                    
                # Global visualization
                plt.figure(figsize=(12, 6))
                for i in range(mu_first.size(0)):
                    std_pred = sigma_first[i].cpu()
                    plt.plot(tgt_first[i].cpu(), color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
                    plt.plot(mu_first[i].cpu(), linewidth=2.0, label='Mean Pred' if i == 0 else None)
                    plt.fill_between(np.arange(mu_first.size(1)), mu_first[i].cpu() - std_pred, mu_first[i].cpu() + std_pred, alpha=0.2, color='red')
                plt.title("All Forecasts: Mean + Predicted Variance"); plt.xlabel("Time step"); plt.ylabel("Forecasted value"); plt.legend(loc='upper right'); plt.tight_layout()
                plt.savefig(_fname("global_overlay")); plt.close()
                visualize = False
        else:
            raise ValueError("reduce must be 'mean' or 'first'")

    # --- Finalize metrics ---
    test_mse = running_mse / len(test_loader.dataset)
    if reduce == "first":
        horizon = mu_first.size(1)
        test_nll  = running_nll  / (len(test_loader.dataset) * horizon); test_crps = running_crps / len(test_loader.dataset)
        test_qpin = running_qpin / len(test_loader.dataset); test_wink = running_wink / len(test_loader.dataset)
    else:
        test_nll = test_crps = test_qpin = test_wink = None
    test_pve_abs = running_pve_abs / len(test_loader.dataset); test_pve_pct = running_pve_pct / len(test_loader.dataset)
    print(f"\n🧪 Test MSE         : {test_mse:.6f}"); print(f"🧪 Test CRPS        : {test_crps:.6f}")
    if reduce == "first":
        print(f"🧪 Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
        print(f"🧪 Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
    if peak_count > 0:
        for k in peak_sum: peak_sum[k] /= peak_count
        pmse_thr = peak_sum.get("PeakPeriodMSE_thr_mean", float("nan"))
        print(f"[PEAK] PVPE_mean={peak_sum['PVPE_mean']:.4f}  PTE_min_mean={peak_sum['PTE_min_mean']:.1f} min  IoU(maxwin)={peak_sum['PeakWindowIoU_mean']:.3f}  IoU(thr)={peak_sum['ThrWindowIoU_mean']:.3f}  PMSE_thr={pmse_thr:.6f}")
    print(f"---------------------------------------------------\n")
    return test_mse, test_nll, test_crps, test_qpin, test_wink


# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    seed        = 42; set_seed(seed); batch_size  = 16; epochs      = 500; lr          = 1e-3
    kl_weight   = 0.01; xprime_dim  = 40; hidden_dim  = 64; latent_dim  = 32; num_layers  = 4; output_len  = 3
    
    encoder_len_weeks = 1 
    
    top_k       = 2; warmup_ep   = 10; device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration for Peak-Weighted Loss
    PEAK_WEIGHT = 10.0  # Assign 10x weight to peak events
    PEAK_Q      = 0.9   # Threshold: Top 10% of values are considered peaks
    print(f"--- Using Peak-Weighted Loss (Weight={PEAK_WEIGHT}, Quantile={PEAK_Q}) ---")


    data_name = "Oncor_load"
    model_name = f"M2OE2_v1_{output_len}hours_Enc{encoder_len_weeks}w"
    
    # Filename includes "peak" to denote the weighted training objective
    BASE_CKPT = f"Oncor_load_{model_name}_peak_best_model.pt"

    if data_name == "Oncor_load":
        times, load, temp, workday, season = get_data_oncor_load_weekly(XFMR="all")
        feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    else:
        raise ValueError(f"Unknown data_name: {data_name}")

    input_dim  = 1; output_dim = 1
    train_data, test_data, _ = process_seq2seq_data(
        feature_dict     = feature_dict, train_ratio      = 0.7, output_len       = output_len,
        encoder_len_weeks = encoder_len_weeks, device           = device)
    n_externals = train_data['X_enc_ext'].shape[-1]; print(f"K_ext (number of external features) = {n_externals}")
    train_loader = make_loader(train_data, batch_size, shuffle=True); test_loader  = make_loader(test_data,  batch_size, shuffle=False)

    # === Pre-train base model if missing ===
    if not os.path.exists(BASE_CKPT):
        print(f"[✗] Base model checkpoint not found. Start pretraining...")
        model_base = VariationalSeq2Seq_meta(
            xprime_dim=xprime_dim, input_dim=input_dim, hidden_size=hidden_dim, latent_size=latent_dim,
            output_len=output_len, n_externals=n_externals, output_dim=output_dim, num_layers=num_layers, dropout=0.1,
        ).to(device)
        optimizer_base = AdamW(model_base.parameters(), lr=lr, weight_decay=1e-4) 
        
        # Initialize training with peak-weighted parameters
        train_model(model_base, train_loader, epochs=epochs, lr=lr, device=device,
                    top_k=top_k, kl_weight=kl_weight, warmup_epochs=warmup_ep,
                    save_path=BASE_CKPT, optimizer=optimizer_base,
                    peak_loss_weight=PEAK_WEIGHT, 
                    peak_threshold_q=PEAK_Q       
                   )
        print(f"[✓] Base model saved to {BASE_CKPT}")
    else:
        print(f"[✓] Base model checkpoint found: {BASE_CKPT}")

    # === LoRA fine-tuning on new data ===
    times, load, temp, workday, season = get_data_oncor_load_weekly(XFMR="176391692")
    feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    train_data, test_data, _ = process_seq2seq_data(
        feature_dict     = feature_dict, train_ratio      = 0.7, output_len       = output_len,
        encoder_len_weeks = encoder_len_weeks, device           = device)
    n_externals = train_data['X_enc_ext'].shape[-1]
    train_loader = make_loader(train_data, batch_size, shuffle=True); test_loader  = make_loader(test_data,  batch_size, shuffle=False)
    
    DATASET_TAG = "ExternalDS"
    plot_data_frames = [] 

    # === Fine-tune: LoRA on head_mu ===
    model_hmu = VariationalSeq2Seq_meta(
        xprime_dim=xprime_dim, input_dim=input_dim, hidden_size=hidden_dim, latent_size=latent_dim,
        output_len=output_len, n_externals=n_externals, output_dim=output_dim, num_layers=num_layers, dropout=0.1
    ).to(device)

    state = torch.load(BASE_CKPT, map_location=device); _ = model_hmu.load_state_dict(state, strict=False) 
    print(f"[✓] model_hmu: Base weights loaded from {BASE_CKPT}")
    test = 8 
    if isinstance(model_hmu.decoder.head_mu, nn.Linear):
        model_hmu.decoder.head_mu = LoRALinear(model_hmu.decoder.head_mu, r=test, alpha=test, lora_dropout=0, train_bias=False)
    print("head_mu class ->", type(model_hmu.decoder.head_mu).__name__)
    if hasattr(model_hmu.decoder, "head_logvar") and isinstance(model_hmu.decoder.head_logvar, nn.Linear):
        model_hmu.decoder.head_logvar = LoRALinear(
            model_hmu.decoder.head_logvar, r=test, alpha=test, lora_dropout=0, train_bias=False)
        print("head_logvar class ->", type(model_hmu.decoder.head_logvar).__name__)
    else:
        raise AttributeError("decoder.head_logvar (variance head) not found or not Linear.")
    
    state = torch.load(BASE_CKPT, map_location=device); _ = model_hmu.load_state_dict(state, strict=False)
    freeze_all(model_hmu); lora_params = collect_lora_params(model_hmu)
    for p in lora_params: p.requires_grad = True
    
    # Filename includes "peak" to signify weighted fine-tuning
    lora_model_path_hmu = f"ExternalDS_headmu_lora_peak_headmu_r{test}.pt"

    optimizer_lora = AdamW(lora_params, lr=1e-3, weight_decay=1e-4) 
    
    # Train LoRA parameters with peak weighting enabled
    train_model(model_hmu, train_loader, epochs=300, lr=lr, device=device,
                top_k=top_k, kl_weight=kl_weight, warmup_epochs=warmup_ep,
                save_path=lora_model_path_hmu,
                optimizer=optimizer_lora,
                peak_loss_weight=PEAK_WEIGHT, 
                peak_threshold_q=PEAK_Q       
               ) 
    
    
    print(f"Reloading best FT model from {lora_model_path_hmu} for evaluation...")
    model_hmu_eval = VariationalSeq2Seq_meta(
        xprime_dim=xprime_dim, input_dim=input_dim, hidden_size=hidden_dim, latent_size=latent_dim,
        output_len=output_len, n_externals=n_externals, output_dim=output_dim, num_layers=num_layers, dropout=0.1
    ).to(device)
    # Re-inject LoRA layers for evaluation
    if isinstance(model_hmu_eval.decoder.head_mu, nn.Linear):
        model_hmu_eval.decoder.head_mu = LoRALinear(model_hmu_eval.decoder.head_mu, r=test, alpha=test, lora_dropout=0.05, train_bias=False)
    if hasattr(model_hmu_eval.decoder, "head_logvar") and isinstance(model_hmu_eval.decoder.head_logvar, nn.Linear):
        model_hmu_eval.decoder.head_logvar = LoRALinear(
            model_hmu_eval.decoder.head_logvar, r=test, alpha=test, lora_dropout=0.05, train_bias=False)
            
    state_ft = torch.load(lora_model_path_hmu, map_location=device)
    model_hmu_eval.load_state_dict(state_ft, strict=False) 
    
    # Evaluate fine-tuned model (explicitly tagged as "peak" version)
    evaluate_model(model_hmu_eval, test_loader, nn.MSELoss(), device, 
                   model_path=lora_model_path_hmu, 
                   data_name=data_name,
                   model_name=f"VAE_LORA_peak_r{test}", 
                   data_export_list=plot_data_frames)


    # --- Save Collected Data ---
    if plot_data_frames:
        print("\n--- Saving Plot Data (VAE) ---")
        final_plot_df = pd.concat(plot_data_frames, ignore_index=True)
        
        # Export comparison data to Excel (peak-specific file)
        xlsx_path_vae = "vae_peak_comparison_data.xlsx" 
        
        try:
            final_plot_df.to_excel(xlsx_path_vae, index=False) 
            print(f"[✓] VAE plot data saved to {xlsx_path_vae}")
        except ImportError:
            print(f"[i] 'openpyxl' not found. Skipping Excel export. (Install with 'pip install openpyxl')")
        except Exception as e:
            print(f"[!] Error saving to Excel: {e}")