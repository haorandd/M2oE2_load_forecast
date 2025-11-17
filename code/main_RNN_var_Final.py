# -*- coding: utf-8 -*-
"""
Probabilistic Recurrent Neural Network (RNN/GRU) Forecasting Framework with Low-Rank Adaptation (LoRA)

This script implements a transfer learning methodology for probabilistic time-series forecasting:
1.  **Source Domain Pre-training**: A base RNN/GRU model is trained on a comprehensive, aggregated dataset (Oncor "all").
2.  **Target Domain Adaptation**: Low-Rank Adaptation (LoRA) is applied to fine-tune the model for a specific target transformer.
3.  **Probabilistic Evaluation**: Performance is quantified using CRPS, NLL, and Winkler Scores.
4.  **Comparative Analysis**: The fine-tuned model is benchmarked against the zero-shot base model.
5.  **Data Aggregation**: Forecast results are exported for centralized comparative plotting.
"""

import os, math, time, random
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from torch.optim import AdamW

# Utility helpers (e.g., sequence reconstruction)
from data_utils import *
# LoRA utilities (assumed from previous script)
from lora_utils import LoRALinear, freeze_all, collect_lora_params

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------------------------------
# Reproducibility Configuration
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across CPU and GPU operations.
    Ensures deterministic behavior for experimental validation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions and Probabilistic Metrics
# ──────────────────────────────────────────────────────────────────────────────
def gaussian_icdf(p, device):
    """
    Computes the inverse cumulative distribution function (probit) for a Gaussian.
    Equation: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1).
    """
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(2*torch.as_tensor(p, device=device) - 1)

def pinball_loss(y, yq, q):
    """
    Calculates the Pinball Loss (Quantile Loss) for a specific quantile q.
    Used to evaluate the calibration of specific parts of the distribution.
    """
    e = y - yq
    return torch.where(e >= 0, q*e, (q-1)*e)

def winkler_score(y, L, U, alpha):
    """
    Computes the Winkler Score to evaluate prediction interval efficiency.
    Penalizes intervals that are too wide or fail to capture the observation.
    """
    width = (U - L)
    below = (L - y).clamp(min=0.0)
    above = (y - U).clamp(min=0.0)
    return width + (2.0/alpha) * (below + above)

def gaussian_nll_loss(mu: torch.Tensor,
                      logvar: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Negative Log-Likelihood (NLL) for Gaussian distributed predictions.
    Serves as the primary loss function for training the probabilistic head.
    """
    nll = 0.5 * (logvar + math.log(2 * math.pi) +
                 (target - mu).pow(2) / logvar.exp())
    return nll.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Data Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_seq2seq_data(
        feature_dict,
        *,
        train_ratio        = 0.7,
        norm_features      = ('load', 'temp'),   # Legacy parameter, kept for compatibility
        output_len         = 24,          
        encoder_len_weeks  = 1,
        decoder_len_weeks  = 1,
        num_in_week        = 168,
        device             = None):
    """
    Transforms raw time-series data into sequence-to-sequence tensors.
    
    The pipeline proceeds as follows:
    1.  **Feature Normalization**: All features (load and external covariates) are scaled to [0, 1] via MinMaxScaler.
    2.  **Sequence Construction**: Sliding windows create encoder inputs (history) and decoder targets (future).
    3.  **Tensor Packing**: Numpy arrays are converted to PyTorch tensors and moved to the computing device.
    """
    # 1) Flatten and scale (normalize ALL features)
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
        raise ValueError(f"Need >= {need_weeks} consecutive weeks, found {n_weeks}.")

    enc_seq_len = encoder_len_weeks * num_in_week
    dec_seq_len = decoder_len_weeks * num_in_week
    L = dec_seq_len - output_len
    if L <= 0:
        raise ValueError("`output_len` must be smaller than decoder sequence length.")

    ext_keys = [k for k in feature_dict.keys() if k != 'load']
    K_ext = len(ext_keys)

    # 2) Build samples (stride = 1 week)
    X_enc_l, X_dec_in_l, Y_dec_target = [], [], []
    X_enc_ext, X_dec_in_ext = [], []

    last_start = n_weeks - need_weeks
    for w in range(last_start + 1):
        enc_start =  w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start =  enc_end
        dec_end   =  dec_start + dec_seq_len

        # Load sequences
        enc_l = processed['load'][enc_start:enc_end]
        dec_full_l = processed['load'][dec_start:dec_end]

        # Encoder/Decoder external covariates
        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)  # [enc_len, K]
            dec_ext = np.stack([processed[k][dec_start: dec_start + L] for k in ext_keys], axis=-1)  # [L, K]
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32)
            dec_ext = np.empty((L, 0), dtype=np.float32)

        # Targets (sliding windows of length output_len across decoder horizon)
        targets = np.stack([dec_full_l[i:i+output_len] for i in range(L+1)], axis=0)  # [L+1, output_len]

        X_enc_l.append(enc_l)
        X_dec_in_l.append(dec_full_l[:L])
        X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext)
        Y_dec_target.append(targets)

    # 3) Pack -> Tensors
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)

    data_tensors = {
        'X_enc_l'      : to_tensor(np.array(X_enc_l)).unsqueeze(-1),         # [B, enc_len, 1]
        'X_enc_ext'    : to_tensor(np.array(X_enc_ext)),                      # [B, enc_len, K_ext]
        'X_dec_in_l'   : to_tensor(np.array(X_dec_in_l)).unsqueeze(-1),       # [B, L, 1]
        'X_dec_in_ext' : to_tensor(np.array(X_dec_in_ext)),                   # [B, L, K_ext]
        'Y_dec_target' : to_tensor(np.array(Y_dec_target)).unsqueeze(-1),     # [B, L+1, output_len, 1]
    }

    for k, v in data_tensors.items():
        print(f"{k:15s} {tuple(v.shape)}")

    # 4) Split into Training and Test sets
    B = data_tensors['X_enc_l'].shape[0]
    split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}
    test_dict  = {k: v[split:] for k, v in data_tensors.items()}
    return train_dict, test_dict, scalers


def make_loader(split_dict, batch_size, shuffle):
    """
    Wraps the dictionary of tensors into a standard PyTorch DataLoader.
    """
    ds = TensorDataset(
        split_dict['X_enc_l'],
        split_dict['X_enc_ext'],
        split_dict['X_dec_in_l'],
        split_dict['X_dec_in_ext'],
        split_dict['Y_dec_target'],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------
# Data Loading Selection
# ---------------------------
def build_feature_dict(data_name: str, XFMR: str = None):
    """
    Selects and loads the appropriate dataset based on the provided configuration.
    Supports specific Oncor transformer selection via the XFMR parameter.
    """
    if data_name == "Building":
        times, load, temp, workday, season = get_data_building_weather_weekly()
        feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    elif data_name == "Spanish":
        times, load, temp, workday, season = get_data_spanish_weekly()
        feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    elif data_name == "Consumption":
        times, load, temp, workday, season = get_data_power_consumption_weekly()
        feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    elif data_name == "Residential":
        times, load, temp, workday, season = get_data_residential_weekly()
        feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    elif data_name == "Solar":
        times, load, temp, workday, season = get_data_solar_weather_weekly()
        feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    elif data_name == "Oncor_load":
        # Select specific transformer or aggregate 'all'
        xfmr_to_load = XFMR if XFMR is not None else "all"
        print(f"[Data] Loading Oncor_load, XFMR='{xfmr_to_load}'")
        times, load, temp, workday, season = get_data_oncor_load_weekly(XFMR=xfmr_to_load)
        feature_dict = {'load': load, 'temp': temp, 'workday': workday, 'season': season}
    elif data_name == "ETTH1":
        times, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT, workday, season_feat = get_data_etth1_weekly()
        feature_dict = {'HUFL': HUFL, 'HULL': HULL, 'MUFL': MUFL, 'MULL': MULL, 'LUFL': LUFL, 'LULL': LULL, 'load': OT, 'workday': workday, 'season': season_feat}
    elif data_name == "ETTH2":
        times, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT, workday, season_feat = get_data_etth2_weekly()
        feature_dict = {'HUFL': HUFL, 'HULL': HULL, 'MUFL': MUFL, 'MULL': MULL, 'LUFL': LUFL, 'LULL': LULL, 'load': OT,
                        'workday': workday, 'season': season_feat}
    elif data_name == "GEFCom2014":
        times, feature_arrays, workday, season_feat, load, feature_names = get_data_GEFCom2014_multi()
        feature_dict = {name: arr for name, arr in zip(feature_names, feature_arrays)}
        feature_dict["load"] = load
    elif data_name == "flores":
        load = get_flores()
        feature_dict = {'load': load}
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    return feature_dict


# ──────────────────────────────────────────────────────────────────────────────
# RNN/GRU Model Architecture
# ──────────────────────────────────────────────────────────────────────────────
class WeekForecastProbRNN(nn.Module):
    """
    Probabilistic Recurrent Architecture (RNN/GRU).

    This module integrates external covariates and temporal history to predict future states.
    Forward pass flow:
    1.  **Input Integration**: Concatenates encoder load sequences with all available external features.
    2.  **Temporal Modeling**: Processes the sequence via stacked GRU or RNN layers.
    3.  **Probabilistic Projection**: A linear head maps the final hidden state to Gaussian parameters (mean and log-variance).
    """
    def __init__(self,
                 input_features: int,      # 1 (load) + K (externals)
                 hidden_size:    int = 128,
                 num_layers:     int = 2,
                 forecast_len:   int = 168,
                 dropout:        float = 0.1,
                 cell_type:      str = "GRU"):   # "GRU" or "RNN"
        super().__init__()

        RNN = nn.GRU if cell_type.upper() == "GRU" else nn.RNN
        self.rnn = RNN(
            input_features, hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Linear head outputs both mean (mu) and log-variance (log sigma^2)
        # This layer serves as the injection point for LoRA adaptation
        self.head = nn.Linear(hidden_size, 2 * forecast_len)
        with torch.no_grad():
            self.head.bias[forecast_len:] = -3.0  # Initialize variance to a low value for stability

    def forward(self, enc_l, enc_ext=None):
        """
        Args:
            enc_l: Encoder load sequence [B, Tenc, 1]
            enc_ext: Encoder external features [B, Tenc, K_ext]
        """
        x = enc_l if (enc_ext is None or enc_ext.numel() == 0) else torch.cat([enc_l, enc_ext], dim=-1)
        # x: [B, Tenc, 1+K]
        _, h_n = self.rnn(x)                # h_n: [num_layers, B, hidden]
        h_last = h_n[-1]                    # [B, hidden]
        out = self.head(h_last)             # [B, 2*forecast_len]
        mu, logvar = out.chunk(2, dim=-1)   # each [B, forecast_len]
        return mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train_prob_model(model, train_loader, *,
                    epochs: int, lr: float,
                    device: torch.device,
                    save_path: str = "Building_RNN_best_model.pt",
                    optimizer: torch.optim.Optimizer = None):
    """
    Executes the gradient-based optimization loop.
    
    This function accommodates both full-model pre-training and parameter-efficient fine-tuning (LoRA).
    If a specific optimizer is provided (e.g., containing only LoRA parameters), it is used; otherwise, a default AdamW optimizer is initialized for all parameters.
    """
    
    # Configure optimizer: Default to full-parameter AdamW if none provided
    if optimizer is None:
        print("[Train] No optimizer provided, creating default AdamW for all params.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        print("[Train] Using provided optimizer (LoRA specific).")

    best  = float("inf")
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            # Unpack batch: enc_l, enc_ext, dec_l, dec_ext, tgt
            enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

            # Reconstruct ground truth load vector for loss calculation
            wk_tgt = torch.stack([reconstruct_sequence(t.squeeze(-1)) for t in tgt])  # [B, 168]

            mu, logvar = model(enc_l, enc_ext)
            loss = gaussian_nll_loss(mu, logvar, wk_tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * enc_l.size(0)

        avg = running / len(train_loader.dataset)
        if avg < best:
            best = avg
            best_epoch = ep
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best @epoch {ep}  NLL {best:.6f}")
        
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | Train NLL {avg:.6f} | Best {best:.6f} (ep {best_epoch})")

    print(f"\n🏁 Training finished. Best epoch {best_epoch} NLL = {best:.6f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CRPS Metric Calculation
# ──────────────────────────────────────────────────────────────────────────────
def crps_gaussian(mu, logvar, target):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for Gaussian predictions.
    Measures the integral squared distance between the cumulative distribution function (CDF) of the forecast and the observation.
    
    Args:
        mu: Predicted mean [B, T]
        logvar: Predicted log-variance [B, T]
        target: True values [B, T]
    """
    std = (0.5 * logvar).exp()          # [B, T]
    z = (target - mu) / std             # [B, T]

    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))        # PDF
    Phi = normal.cdf(z)                        # CDF

    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation and Visualization Engine
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_rnn_model(model,
                       test_loader,
                       device,
                       model_path="Building_RNN_best_model.pt",
                       visualize=True,
                       n_vis_samples=5,
                       data_name="data",
                       model_name="model",
                       quantiles=(0.1, 0.5, 0.9),
                       alpha=0.1,
                       data_export_list=None):
    """
    Performs comprehensive model evaluation.
    
    Process:
    1.  **Metric Calculation**: Computes MSE, NLL, CRPS, Quantile Loss, and Winkler Scores.
    2.  **Visualization**: Generates forecast plots (individual and global overlays).
    3.  **Data Export**: Aggregates prediction data for external comparative analysis.
    """
    print(f"--- Evaluating model '{model_name}' on data '{data_name}' ---")
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()

    mse_fn = torch.nn.MSELoss(reduction="mean")
    running_mse = 0.0
    running_nll = 0.0
    running_crps = 0.0
    running_qpin = 0.0
    running_wink = 0.0

    os.makedirs("result", exist_ok=True)
    
    # Helper for consistent file naming
    def _fname(tag, i=None):
        if i is None:
            return f"./result/{model_name}_{tag}.pdf"
        return f"./result/{model_name}_{tag}_{i}.pdf"

    for batch in test_loader:
        enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

        mu, logvar = model(enc_l, enc_ext)
        B, H = mu.shape
        sigma = logvar.exp().sqrt()
        wk_tgt = torch.stack([reconstruct_sequence(t.squeeze(-1)) for t in tgt])

        # ... (Metrics calculation) ...
        running_mse += mse_fn(mu, wk_tgt).item() * B
        nll = 0.5 * (logvar + torch.log(torch.tensor(2 * np.pi, device=logvar.device)) + (wk_tgt - mu).pow(2) / logvar.exp())
        running_nll += nll.sum().item()
        crps = crps_gaussian(mu, logvar, wk_tgt)
        running_crps += crps.item() * B
        q_losses = []
        for q in quantiles:
            zq = gaussian_icdf(q, device=mu.device); yq = mu + sigma * zq
            ql = pinball_loss(wk_tgt, yq, q).mean(); q_losses.append(ql)
        qpin_mean = torch.stack(q_losses).mean(); running_qpin += qpin_mean.item() * B
        z = gaussian_icdf(1.0 - alpha/2.0, device=mu.device)
        L = mu - z * sigma; U = mu + z * sigma
        ws = winkler_score(wk_tgt, L, U, alpha).mean(); running_wink += ws.item() * B


        # ---- Visualisation -----------------------------------------------
        if visualize:
            x_axis = np.arange(H)
            
            # --- Plot 1: Standard Forecast ---
            for i in range(min(n_vis_samples, B)):
                std_pred = sigma[i].cpu()
                plt.figure(figsize=(4, 2))
                plt.plot(wk_tgt[i].cpu()[:H-2], '--', color='red', label='True')
                plt.plot(mu[i].cpu()[:H-2], color='blue', alpha=0.6, label='Mean Pred')
                plt.fill_between(x_axis[:H-2], (mu[i].cpu() - std_pred)[:H-2], (mu[i].cpu() + std_pred)[:H-2], color='blue', alpha=0.1, label='±1 $\sigma$')
                plt.tight_layout(); plt.ylim(0, 1); plt.yticks([0, 0.5, 1], fontsize=14); plt.xticks(fontsize=14)
                plt.savefig(_fname("sample", i)) 
                plt.close()

            # --- Plot 2: Forecast with Historical Context ---
            for i in range(min(n_vis_samples, B)):
                std_pred = sigma[i].cpu()
                mu_i = mu[i].cpu()
                y_true_i = wk_tgt[i].cpu()
                hist_i = enc_l[i].cpu().squeeze(-1) 

                Lh = len(hist_i); H_pred  = len(mu_i)
                x_hist = np.arange(Lh); x_fore = np.arange(Lh, Lh + H_pred)

                plt.figure(figsize=(10, 2.5))
                plt.plot(x_hist, hist_i, color='black', linewidth=1.5, label='History')
                plt.plot(x_fore, y_true_i, '--', color='red', linewidth=1.5, label='True')
                plt.plot(x_fore, mu_i, color='blue', alpha=0.8, linewidth=1.5, label='Mean Pred')
                plt.fill_between(x_fore, mu_i - std_pred, mu_i + std_pred, color='blue', alpha=0.1, label='±1 $\sigma$ (pred.)')
                plt.axvline(Lh - 1, color='grey', linestyle='--', alpha=0.6)
                plt.xlim(0, Lh + H_pred); plt.ylim(0, 1); plt.tight_layout(); plt.yticks([0, 0.5, 1], fontsize=14); plt.xticks(fontsize=14); plt.legend()

                # --- Data Export ---
                if data_export_list is not None:
                    df_hist = pd.DataFrame({'time_step': x_hist, 'value': hist_i.numpy(), 'value_type': 'history'})
                    df_true = pd.DataFrame({'time_step': x_fore, 'value': y_true_i.numpy(), 'value_type': 'true'})
                    df_pred = pd.DataFrame({'time_step': x_fore, 'value': mu_i.numpy(), 'value_type': 'pred_mean'})
                    df_std = pd.DataFrame({'time_step': x_fore, 'value': std_pred.numpy(), 'value_type': 'pred_std'})
                    df_sample = pd.concat([df_hist, df_true, df_pred, df_std])
                    df_sample['model_name'] = model_name
                    df_sample['sample_index'] = i
                    data_export_list.append(df_sample)
                    print(f"[Data Export] Saved sample {i} for model {model_name}")
                
                plt.savefig(_fname("with_hist", i)) 
                plt.close()


            # --- Plot 3: Global Overlay ---
            plt.figure(figsize=(12, 6))
            for i in range(B):
                std_pred = sigma[i].cpu()
                plt.plot(wk_tgt[i].cpu(), '--', color='grey', linewidth=0.8, alpha=0.4)
                plt.plot(mu[i].cpu(), linewidth=2.0, color='blue', label='Mean Pred' if i == 0 else None)
                plt.fill_between(x_axis, mu[i].cpu() - std_pred, mu[i].cpu() + std_pred, alpha=0.2, color='red')
            plt.xlabel("Time step");  plt.ylabel("Load"); plt.title("All Forecasts • Mean + Predicted Variance"); plt.legend(loc='upper right'); plt.tight_layout()
            plt.savefig(_fname("global_overlay")) 
            plt.close()
            
            visualize = False # Only run once

    # ---- Final Metrics Aggregation ----------------------------------------------------
    num_pts  = len(test_loader.dataset) * H
    test_mse   = running_mse  / len(test_loader.dataset)
    test_nll   = running_nll  / num_pts
    test_crps  = running_crps / len(test_loader.dataset)
    test_qpin  = running_qpin / len(test_loader.dataset)
    test_wink  = running_wink / len(test_loader.dataset)

    print(f"\nTest MSE         : {test_mse:.6f}")
    print(f"Test CRPS        : {test_crps:.6f}")
    print(f"Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
    print(f"Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
    print(f"---------------------------------------------------\n")

    return test_mse, test_nll, test_crps, test_qpin, test_wink


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution Block
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(42)
    
    # --- Shared Hyperparameters ---
    batch_size   = 32
    epochs_pretrain = 200 
    epochs_finetune = 50  
    lr_pretrain  = 1e-3
    lr_finetune  = 1e-4  
    output_len         = 3
    
    # Aligned with VAE settings to ensure consistent comparison
    encoder_len_weeks  = 1           
    
    decoder_len_weeks  = 1
    
    # Model architecture params
    hidden_size  = 128
    num_layers   = 2
    forecast_len = 168
    dropout      = 0.1
    cell_type    = "GRU"
    
    # LoRA params
    lora_r       = 8
    lora_alpha   = 8
    
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data and Model Paths ---
    data_name    = "Oncor_load"
    # Architecture tag generation
    model_arch   = f"RNN_{cell_type}_L{num_layers}_H{hidden_size}_Enc{encoder_len_weeks}w"
    
    # Base model: trained on "all"
    BASE_MODEL_TAG = f"{data_name}_ALL_{model_arch}"
    BASE_CKPT    = f"{BASE_MODEL_TAG}_best_model.pt"
    
    # Target (fine-tune) data
    TARGET_XFMR  = "176391692"
    TARGET_TAG   = f"{data_name}_XFMR_{TARGET_XFMR}"

    # LoRA model: base model + LoRA tuned on target
    LORA_MODEL_TAG = f"{TARGET_TAG}_LORA_r{lora_r}"
    LORA_CKPT    = f"{LORA_MODEL_TAG}_best_model.pt"


    # ==========================================================================
    # (A) Source Domain Pre-training
    # ==========================================================================
    
    # Load base dataset (Oncor "all")
    print(f"--- Loading Base Data ({data_name} XFMR='all') ---")
    feature_dict_base = build_feature_dict(data_name, XFMR="all")
    train_data_base, test_data_base, _ = process_seq2seq_data(
        feature_dict       = feature_dict_base,
        train_ratio        = 0.7,
        output_len         = output_len,
        encoder_len_weeks  = encoder_len_weeks,
        decoder_len_weeks  = decoder_len_weeks,
        device             = device)

    n_externals = train_data_base['X_enc_ext'].shape[-1]
    print(f"K_ext (number of external features) = {n_externals}")
    
    input_features = 1 + n_externals

    # Check for existing base model checkpoint
    if not os.path.isfile(BASE_CKPT):
        print(f"[✗] Base model checkpoint not found. Start pretraining...")
        
        train_loader_base = make_loader(train_data_base, batch_size, shuffle=True)
        
        model_base = WeekForecastProbRNN(
            input_features=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_len=forecast_len,
            dropout=dropout,
            cell_type=cell_type
        ).to(device)

        train_prob_model(model_base, train_loader_base, epochs=epochs_pretrain, lr=lr_pretrain,
                         device=device, save_path=BASE_CKPT)
        print(f"[✓] Base model pretraining complete. Saved to {BASE_CKPT}")
        
        # Clean up memory
        del model_base, train_loader_base
        torch.cuda.empty_cache()
    else:
        print(f"[✓] Found existing base model: {BASE_CKPT}")
    
    # Clean up base data dicts
    del train_data_base, test_data_base, feature_dict_base
    torch.cuda.empty_cache()


    # ==========================================================================
    # (B) Target Domain Data Loading
    # ==========================================================================
    print(f"--- Loading Target Data ({data_name} XFMR='{TARGET_XFMR}') ---")
    feature_dict_target = build_feature_dict(data_name, XFMR=TARGET_XFMR)
    train_data_target, test_data_target, _ = process_seq2seq_data(
        feature_dict       = feature_dict_target,
        train_ratio        = 0.7,
        output_len         = output_len,
        encoder_len_weeks  = encoder_len_weeks,
        decoder_len_weeks  = decoder_len_weeks,
        device             = device)

    # Create loaders for fine-tuning and evaluation
    train_loader_target = make_loader(train_data_target, batch_size, shuffle=True)
    test_loader_target  = make_loader(test_data_target,  batch_size, shuffle=False)
    
    del train_data_target, test_data_target, feature_dict_target
    torch.cuda.empty_cache()


    # ==========================================================================
    # (C) Low-Rank Adaptation (LoRA) Fine-Tuning
    # ==========================================================================
    print(f"--- Starting LoRA Fine-Tuning ---")
    
    # 1. Initialize model architecture
    model_lora = WeekForecastProbRNN(
        input_features=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_len=forecast_len,
        dropout=dropout,
        cell_type=cell_type
    ).to(device)

    # 2. Inject LoRA module into the output projection head
    if isinstance(model_lora.head, nn.Linear):
        model_lora.head = LoRALinear(
            model_lora.head, r=lora_r, alpha=lora_alpha, lora_dropout=0.05, train_bias=False
        )
    print(f"Successfully injected LoRA into model.head.")
    print("head class ->", type(model_lora.head).__name__)

    # 3. Load pre-trained base weights (strict=False allows loading despite LoRA layer modifications)
    state = torch.load(BASE_CKPT, map_location=device)
    _ = model_lora.load_state_dict(state, strict=False)
    print(f"[✓] LoRA Model: Base weights loaded from {BASE_CKPT}")

    # 4. Freeze backbone and activate LoRA parameters
    freeze_all(model_lora)
    lora_params = collect_lora_params(model_lora)
    for p in lora_params: p.requires_grad = True
    
    num_lora_params = sum(p.numel() for p in lora_params)
    print(f"[Train] Training {num_lora_params} LoRA parameters.")

    # 5. Configure LoRA-specific optimizer
    optimizer_lora = AdamW(lora_params, lr=lr_finetune, weight_decay=1e-4)

    # 6. Execute Fine-tuning
    train_prob_model(model_lora, train_loader_target, 
                     epochs=150,
                     lr=lr_finetune, 
                     device=device, 
                     save_path=LORA_CKPT,
                     optimizer=optimizer_lora) 
    
    print(f"[✓] LoRA fine-tuning complete. Saved to {LORA_CKPT}")


    # ==========================================================================
    # (D) Comparative Evaluation
    # ==========================================================================
    
    # Initialize container for aggregate plot data
    plot_data_frames = []


    # --- Evaluation A: LoRA-Adapted Model ---
    t1 = time.time()
    
    evaluate_rnn_model(model_lora,
                       test_loader_target,
                       device,
                       model_path=LORA_CKPT,
                       visualize=True,
                       data_name=TARGET_TAG,
                       model_name=f"RNN_{cell_type}_LORA_r{lora_r}", 
                       data_export_list=plot_data_frames            
                       )
    t2 = time.time()
    print("LoRA Model Eval wall time (s):", t2 - t1)
    
    del model_lora 
    torch.cuda.empty_cache()

    # --- Evaluation B: Zero-Shot Base Model ---
    print(f"--- Starting Base Model (Zero-Shot) Evaluation ---")
    
    # Initialize clean model instance for baseline comparison
    model_base_eval = WeekForecastProbRNN(
        input_features=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_len=forecast_len,
        dropout=dropout,
        cell_type=cell_type
    ).to(device)

    t1 = time.time()
    
    evaluate_rnn_model(model_base_eval,
                       test_loader_target,
                       device,
                       model_path=BASE_CKPT, 
                       visualize=True,
                       data_name=TARGET_TAG, 
                       model_name=f"RNN_{cell_type}_BASE_ZeroShot", 
                       data_export_list=plot_data_frames           
                       )
    t2 = time.time()
    print("Base Model (Zero-Shot) Eval wall time (s):", t2 - t1)


    # --- Data Persistence ---
    # Export aggregated results for comparative plotting
    if plot_data_frames:
        print("\n--- Saving Plot Data ---")
        final_plot_df = pd.concat(plot_data_frames, ignore_index=True)
        
        csv_path = "rnn_vae_comparison_data.csv"
        xlsx_path = "rnn_vae_comparison_data.xlsx"
        
        # Save to CSV (mode='w' ensures clean write)
        final_plot_df.to_csv(csv_path, index=False, mode='w')
        print(f"[✓] Plot data saved to {csv_path}")
        
        # Optional Excel export
        try:
            final_plot_df.to_excel(xlsx_path, index=False)
            print(f"[✓] Plot data saved to {xlsx_path}")
        except ImportError:
            print(f"[i] 'openpyxl' not found. Skipping Excel export. (Install with 'pip install openpyxl')")