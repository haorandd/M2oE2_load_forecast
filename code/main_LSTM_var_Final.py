# -*- coding: utf-8 -*-
"""
Probabilistic LSTM Forecasting Framework with Low-Rank Adaptation (LoRA)

This script implements a Long Short-Term Memory (LSTM) network for probabilistic time-series forecasting.
The workflow demonstrates a transfer learning approach:
1.  **Pre-training**: The model is trained on a comprehensive source domain (aggregated Oncor data).
2.  **Adaptation**: Low-Rank Adaptation (LoRA) is applied to fine-tune the model for a specific target transformer.
3.  **Evaluation**: The performance is assessed using CRPS, NLL, and Winkler Score metrics.
4.  **Comparison**: We compare the fine-tuned model against the zero-shot base model.
5.  **Data Export**: Results are appended to a shared repository for aggregate analysis.
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



# your helpers (e.g., reconstruct_sequence)

from data_utils import *

# lora utils

from lora_utils import LoRALinear, freeze_all, collect_lora_params



# needed by your provided data block

from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------------------------------
# Reproducibility Configuration
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


# ------------------------------------------------------------------------------
# Loss Functions and Metrics
# ------------------------------------------------------------------------------
def crps_gaussian(mu, logvar, target):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for Gaussian predictions.
    
    This metric assesses the accuracy of the cumulative distribution function.
    
    Args:
        mu: Predicted mean.
        logvar: Predicted log-variance.
        target: True target values.
        
    Returns:
        Mean CRPS score (lower is better).
    """
    std = (0.5 * logvar).exp()
    z = (target - mu) / std
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))        # PDF
    Phi = normal.cdf(z)                        # CDF
    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()

def gaussian_icdf(p, device):
    """
    Computes the inverse cumulative distribution function (probit) for a Gaussian.
    Equation: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1).
    """
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(2*torch.as_tensor(p, device=device) - 1)

def pinball_loss(y, yq, q):
    """
    Calculates the Pinball Loss (Quantile Loss).
    This measures the accuracy of a specific quantile prediction.
    """
    e = y - yq
    return torch.where(e >= 0, q*e, (q-1)*e)

def winkler_score(y, L, U, alpha):
    """
    Computes the Winkler Score for prediction intervals.
    It penalizes intervals that are too wide or fail to capture the observation.
    """
    width = (U - L)
    below = (L - y).clamp(min=0.0)
    above = (y - U).clamp(min=0.0)
    return width + (2.0/alpha) * (below + above)

def gaussian_nll_loss(mu: torch.Tensor,
                      logvar: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Negative Log-Likelihood (NLL) for a Gaussian distribution.
    """
    nll = 0.5 * (logvar + math.log(2 * math.pi) +
                 (target - mu).pow(2) / logvar.exp())
    return nll.mean()


# ------------------------------------------------------------------------------
# Data Preprocessing Pipeline
# ------------------------------------------------------------------------------
def process_seq2seq_data(
        feature_dict,
        *,
        train_ratio        = 0.7,
        norm_features      = ('load', 'temp'),
        output_len         = 24,
        encoder_len_weeks  = 1,
        decoder_len_weeks  = 1,
        num_in_week        = 168,
        device             = None):
    """
    Transforms raw time-series data into sequence-to-sequence tensors.
    
    The process involves three main steps:
    1.  Feature Normalization: All input features are scaled to the [0, 1] range.
    2.  Sequence Construction: Sliding windows create encoder inputs and decoder targets.
    3.  Tensor Packing: Numpy arrays are converted to PyTorch tensors on the specified device.
    """
    
    # 1. Flatten and normalize features
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

    # 2. Build samples using a sliding window (stride = 1 week)
    X_enc_l, X_dec_in_l, Y_dec_target = [], [], []
    X_enc_ext, X_dec_in_ext = [], []

    last_start = n_weeks - need_weeks
    for w in range(last_start + 1):
        enc_start =  w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start =  enc_end
        dec_end   =  dec_start + dec_seq_len

        # Extract load sequences
        enc_l = processed['load'][enc_start:enc_end]
        dec_full_l = processed['load'][dec_start:dec_end]

        # Extract external covariates
        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)
            dec_ext = np.stack([processed[k][dec_start: dec_start + L] for k in ext_keys], axis=-1)
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32)
            dec_ext = np.empty((L, 0), dtype=np.float32)

        targets = np.stack([dec_full_l[i:i+output_len] for i in range(L+1)], axis=0)

        X_enc_l.append(enc_l)
        X_dec_in_l.append(dec_full_l[:L])
        X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext)
        Y_dec_target.append(targets)

    # 3. Convert to Tensors
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

    # 4. Split into Training and Test sets
    B = data_tensors['X_enc_l'].shape[0]
    split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}
    test_dict  = {k: v[split:] for k, v in data_tensors.items()}
    return train_dict, test_dict, scalers


def make_loader(split_dict, batch_size, shuffle):
    """
    Wraps dictionary tensors into a PyTorch DataLoader.
    """
    ds = TensorDataset(
        split_dict['X_enc_l'],
        split_dict['X_enc_ext'],
        split_dict['X_dec_in_l'],
        split_dict['X_dec_in_ext'],
        split_dict['Y_dec_target'],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ------------------------------------------------------------------------------
# Data Loading Selection
# ------------------------------------------------------------------------------
def build_feature_dict(data_name: str, XFMR: str = None):
    """
    Selects and loads the appropriate dataset based on the provided name.
    It handles various data sources, including specific Oncor transformers.
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
        # Handle specific transformer selection
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


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
class WeekForecastProbLSTM(nn.Module):
    """
    Probabilistic LSTM Architecture.

    This model processes concatenated load and external features using an LSTM.
    The final hidden state is projected by a linear head to output probabilistic parameters.
    
    Forward pass:
    1.  Input Concatenation: Load + External features -> [B, T, 1+K].
    2.  Sequence Modeling: LSTM processes the temporal sequence.
    3.  Output Projection: Linear head predicts mean (mu) and log-variance (logvar).
    """
    def __init__(self,
                 input_features: int,
                 hidden_size:    int = 128,
                 num_layers:     int = 2,
                 forecast_len:   int = 168,
                 dropout:        float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_features, hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Head outputs both mu and log(sigma^2) for the forecast horizon
        self.head = nn.Linear(hidden_size, 2 * forecast_len)
        with torch.no_grad():
            self.head.bias[forecast_len:] = -3.0  # Initialize variance to a small value

    def forward(self, enc_l, enc_ext=None):
        """
        Performs the forward pass.
        Args:
            enc_l: Encoder load sequence [B, Tenc, 1].
            enc_ext: Encoder external features [B, Tenc, K_ext].
        Returns:
            mu, logvar: Predicted parameters for the Gaussian distribution.
        """
        x = enc_l if (enc_ext is None or enc_ext.numel() == 0) else torch.cat([enc_l, enc_ext], dim=-1)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        out = self.head(h_last)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


# ------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------
def train_prob_lstm(model, train_loader, *,
                    epochs: int, lr: float,
                    device: torch.device,
                    save_path: str = "Building_LSTM_best_model.pt",
                    optimizer: torch.optim.Optimizer = None):
    """
    Executes the training loop for the probabilistic model.
    
    This function supports both full training and partial fine-tuning.
    If an optimizer is provided, it utilizes it (e.g., for LoRA parameters only).
    Otherwise, it initializes a default AdamW optimizer for all model parameters.
    """
    
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
            enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

            # Reconstruct ground truth load vector
            wk_tgt = torch.stack([
                reconstruct_sequence(t.squeeze(-1))
                for t in tgt
            ])

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
            print(f"Saved best @epoch {ep}  NLL {best:.6f}")
        
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | Train NLL {avg:.6f} | Best {best:.6f} (ep {best_epoch})")

    print(f"\nTraining finished. Best epoch {best_epoch} NLL = {best:.6f}")
    return model


# ------------------------------------------------------------------------------
# Evaluation Engine
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_lstm_model(model,
                        test_loader,
                        device,
                        model_path="Building_LSTM_best_model.pt",
                        visualize=True,
                        n_vis_samples=5,
                        data_name="data",
                        model_name="model",
                        quantiles=(0.1, 0.5, 0.9),
                        alpha=0.1,
                        data_export_list=None):
    """
    Evaluates the model on the test dataset using multiple probabilistic metrics.
    
    The evaluation process includes:
    1.  Metric Calculation: Computes MSE, NLL, CRPS, Quantile Loss, and Winkler Score.
    2.  Visualization: Generates plots for individual samples and global overlays.
    3.  Data Export: Appends prediction data to a list for aggregate analysis.
    """
    print(f"--- Evaluating model '{model_name}' on data '{data_name}' ---")
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()

    mse_fn = torch.nn.MSELoss(reduction="mean")
    running_mse, running_nll, running_crps = 0.0, 0.0, 0.0
    running_qpin, running_wink = 0.0, 0.0

    os.makedirs("result", exist_ok=True)
    
    def _fname(tag, i=None):
        if i is None:
            return f"./result/{model_name}_{tag}.pdf"
        return f"./result/{model_name}_{tag}_{i}.pdf"

    for batch in test_loader:
        enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

        # Forward pass
        mu, logvar = model(enc_l, enc_ext)
        B, horizon = mu.size()
        sigma = logvar.exp().sqrt()

        # Reconstruct ground truth
        wk_tgt = torch.stack(
            [reconstruct_sequence(t.squeeze(-1)) for t in tgt]
        )

        # Base metrics
        running_mse += mse_fn(mu, wk_tgt).item() * B
        nll = 0.5 * (logvar + torch.log(torch.tensor(2*np.pi, device=logvar.device))
                     + (wk_tgt - mu).pow(2) / logvar.exp())
        running_nll += nll.sum().item()
        crps = crps_gaussian(mu, logvar, wk_tgt)
        running_crps += crps.item() * B

        # Quantile (pinball) loss
        q_losses = []
        for q in quantiles:
            zq = gaussian_icdf(q, device=mu.device); yq = mu + sigma * zq
            ql = pinball_loss(wk_tgt, yq, q).mean(); q_losses.append(ql)
        qpin_mean = torch.stack(q_losses).mean(); running_qpin += qpin_mean.item() * B

        # Winkler score for (1-alpha) Prediction Interval
        z = gaussian_icdf(1.0 - alpha/2.0, device=mu.device)
        L = mu - z * sigma; U = mu + z * sigma
        ws = winkler_score(wk_tgt, L, U, alpha).mean()
        running_wink += ws.item() * B

        # Visualisation and Data Export
        if visualize:
            x_axis = np.arange(horizon)

            # 1. Plot Sample
            for i in range(min(n_vis_samples, B)):
                std_pred = sigma[i].cpu()
                plt.figure(figsize=(4, 2))
                plt.plot(wk_tgt[i].cpu()[:166], '--', color='red',  label='True')
                plt.plot(mu[i].cpu()[:166],      color='blue', alpha=0.6, label='Mean Pred')
                plt.fill_between(x_axis[:166], (mu[i].cpu() - std_pred)[:166], (mu[i].cpu() + std_pred)[:166], color='blue', alpha=0.1, label='±1 σ (pred.)')
                plt.tight_layout(); plt.ylim(0, 1); plt.yticks([0, 0.5, 1], fontsize=14); plt.xticks(fontsize=14)
                plt.savefig(_fname("sample", i))
                plt.close()

            # 2. Plot Sample with History Context
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

                # Export data for subsequent analysis
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


            # 3. Plot Global Overlay
            plt.figure(figsize=(12, 6))
            for i in range(B):
                std_pred = sigma[i].cpu()
                plt.plot(wk_tgt[i].cpu(), '--', color='grey', linewidth=0.8, alpha=0.4)
                plt.plot(mu[i].cpu(), linewidth=2.0, color='blue',
                         label='Mean Pred' if i == 0 else None)
                plt.fill_between(x_axis, mu[i].cpu() - std_pred, mu[i].cpu() + std_pred,
                                 alpha=0.2, color='red')
            plt.xlabel("Time step");  plt.ylabel("Load")
            plt.title("All Forecasts: Mean + Predicted Variance")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(_fname("global_overlay"))
            visualize = False

    # Aggregate Metrics
    num_pts   = len(test_loader.dataset) * horizon
    test_mse  = running_mse  / len(test_loader.dataset)
    test_nll  = running_nll  / num_pts
    test_crps = running_crps / len(test_loader.dataset)
    test_qpin = running_qpin / len(test_loader.dataset)
    test_wink = running_wink / len(test_loader.dataset)

    print(f"\nTest MSE         : {test_mse:.6f}")
    print(f"Test CRPS        : {test_crps:.6f}")
    print(f"Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
    print(f"Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
    print(f"---------------------------------------------------\n")

    return test_mse, test_nll, test_crps, test_qpin, test_wink


# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)

    # 1. Hyperparameter Configuration
    batch_size   = 32
    epochs_pretrain = 200
    epochs_finetune = 50
    lr_pretrain  = 1e-3
    lr_finetune  = 1e-4
    output_len         = 3
    
    encoder_len_weeks  = 1
    decoder_len_weeks  = 1

    # LSTM Architecture parameters
    hidden_size  = 128
    num_layers   = 2
    forecast_len = 168
    dropout      = 0.1
    
    # LoRA configuration
    lora_r       = 8
    lora_alpha   = 8
    
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths and model tags
    data_name    = "Oncor_load"
    model_arch   = f"LSTM_L{num_layers}_H{hidden_size}_Enc{encoder_len_weeks}w"
    
    BASE_MODEL_TAG = f"{data_name}_ALL_{model_arch}"
    BASE_CKPT    = f"{BASE_MODEL_TAG}_best_model.pt"
    
    TARGET_XFMR  = "176391692"
    TARGET_TAG   = f"{data_name}_XFMR_{TARGET_XFMR}"

    LORA_MODEL_TAG = f"{TARGET_TAG}_LORA_r{lora_r}"
    LORA_CKPT    = f"{LORA_MODEL_TAG}_best_model.pt"


    # 2. Source Domain Pre-training
    # We first train the base model on the aggregated dataset ("all").
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

    if not os.path.isfile(BASE_CKPT):
        print(f"Base model checkpoint not found. Initiating pretraining...")
        
        train_loader_base = make_loader(train_data_base, batch_size, shuffle=True)
        
        model_base = WeekForecastProbLSTM(
            input_features=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_len=forecast_len,
            dropout=dropout
        ).to(device)

        train_prob_lstm(model_base, train_loader_base, epochs=epochs_pretrain, lr=lr_pretrain,
                         device=device, save_path=BASE_CKPT)
        print(f"Base model pretraining complete. Saved to {BASE_CKPT}")
        
        del model_base, train_loader_base
        torch.cuda.empty_cache()
    else:
        print(f"Found existing base model: {BASE_CKPT}")
    
    del train_data_base, test_data_base, feature_dict_base
    torch.cuda.empty_cache()


    # 3. Target Domain Adaptation setup
    # We load the specific target data for fine-tuning and evaluation.
    print(f"--- Loading Target Data ({data_name} XFMR='{TARGET_XFMR}') ---")
    feature_dict_target = build_feature_dict(data_name, XFMR=TARGET_XFMR)
    train_data_target, test_data_target, _ = process_seq2seq_data(
        feature_dict       = feature_dict_target,
        train_ratio        = 0.7,
        output_len         = output_len,
        encoder_len_weeks  = encoder_len_weeks,
        decoder_len_weeks  = decoder_len_weeks,
        device             = device)
    
    train_loader_target = make_loader(train_data_target, batch_size, shuffle=True)
    test_loader_target  = make_loader(test_data_target,  batch_size, shuffle=False)
    
    del train_data_target, test_data_target, feature_dict_target
    torch.cuda.empty_cache()


    # 4. LoRA Fine-Tuning
    # We initialize the model, inject LoRA into the output head, and fine-tune on the target.
    print(f"--- Starting LoRA Fine-Tuning ---")
    
    model_lora = WeekForecastProbLSTM(
        input_features=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_len=forecast_len,
        dropout=dropout
    ).to(device)

    # Inject LoRA into the output layer
    if isinstance(model_lora.head, nn.Linear):
        model_lora.head = LoRALinear(
            model_lora.head, r=lora_r, alpha=lora_alpha, lora_dropout=0.05, train_bias=False
        )
    print(f"LoRA injection successful. Head type: {type(model_lora.head).__name__}")

    # Load base weights (strict=False allows loading weights despite LoRA layer changes)
    state = torch.load(BASE_CKPT, map_location=device)
    _ = model_lora.load_state_dict(state, strict=False)
    print(f"LoRA Model: Base weights loaded from {BASE_CKPT}")

    # Freeze base parameters and activate LoRA gradients
    freeze_all(model_lora)
    lora_params = collect_lora_params(model_lora)
    for p in lora_params: p.requires_grad = True
    
    num_lora_params = sum(p.numel() for p in lora_params)
    print(f"Training {num_lora_params} LoRA parameters.")

    optimizer_lora = AdamW(lora_params, lr=lr_finetune, weight_decay=1e-4)

    train_prob_lstm(model_lora, train_loader_target,
                     epochs=150,
                     lr=lr_finetune,
                     device=device, 
                     save_path=LORA_CKPT,
                     optimizer=optimizer_lora)
    
    print(f"LoRA fine-tuning complete. Saved to {LORA_CKPT}")


    # 5. Comparative Evaluation
    # We compare the fine-tuned LoRA model against the zero-shot Base model.
    plot_data_frames = []

    # Evaluation A: LoRA-tuned Model
    t1 = time.time()
    evaluate_lstm_model(model_lora,
                       test_loader_target,
                       device,
                       model_path=LORA_CKPT,
                       visualize=True,
                       data_name=TARGET_TAG,
                       model_name=f"LSTM_LORA_r{lora_r}",
                       data_export_list=plot_data_frames
                       )
    t2 = time.time()
    print("LoRA Model Eval wall time (s):", t2 - t1)
    
    del model_lora
    torch.cuda.empty_cache()

    # Evaluation B: Base Model (Zero-Shot)
    print(f"--- Starting Base Model (Zero-Shot) Evaluation ---")
    model_base_eval = WeekForecastProbLSTM(
        input_features=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_len=forecast_len,
        dropout=dropout
    ).to(device)

    t1 = time.time()
    evaluate_lstm_model(model_base_eval,
                       test_loader_target,
                       device,
                       model_path=BASE_CKPT, 
                       visualize=True,
                       data_name=TARGET_TAG, 
                       model_name="LSTM_BASE_ZeroShot",
                       data_export_list=plot_data_frames
                       )
    t2 = time.time()
    print("Base Model (Zero-Shot) Eval wall time (s):", t2 - t1)


    # 6. Result Export
    # Consolidated prediction data is saved for external plotting and analysis.
    if plot_data_frames:
        print("\n--- Saving Plot Data ---")
        final_plot_df = pd.concat(plot_data_frames, ignore_index=True)
        
        # Append to shared CSV for model comparison
        csv_path = "rnn_vae_comparison_data.csv"
        file_exists = os.path.isfile(csv_path)
        
        final_plot_df.to_csv(csv_path, index=False, 
                             mode='a',
                             header=not file_exists)
        print(f"Plot data appended to {csv_path}")
        
        # Save specific Excel file
        xlsx_path_lstm = "lstm_comparison_data.xlsx"
        try:
            final_plot_df.to_excel(xlsx_path_lstm, index=False)
            print(f"LSTM plot data saved to {xlsx_path_lstm}")
        except ImportError:
            print(f"Note: 'openpyxl' not found. Skipping Excel export.")