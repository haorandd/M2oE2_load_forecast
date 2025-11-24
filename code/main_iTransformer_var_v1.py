from utils_iTransformer import *
import math, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# (keep the rest of your original imports)
from data_utils import *
from model import *
import numpy as np, random, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import math
from torch.distributions.normal import Normal
import os, math, time, random

# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Helper losses / metrics
# ──────────────────────────────────────────────────────────────────────────────
def gaussian_icdf(p, device):
    # Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        2 * torch.as_tensor(p, device=device) - 1
    )

def pinball_loss(y, yq, q):
    # y, yq: same shape; q in (0,1)
    e = y - yq
    return torch.where(e >= 0, q*e, (q-1)*e)

def winkler_score(y, L, U, alpha):
    # elementwise Winkler; lower is better
    width = (U - L)
    below = (L - y).clamp(min=0.0)
    above = (y - U).clamp(min=0.0)
    return width + (2.0/alpha) * (below + above)

def gaussian_nll_loss(mu: torch.Tensor,
                      logvar: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    mu, logvar, target : [B, 168]
    Returns scalar mean negative-log-likelihood.
    """
    nll = 0.5 * (logvar + math.log(2 * math.pi) +
                 (target - mu).pow(2) / logvar.exp())
    return nll.mean()

def crps_gaussian(mu, logvar, target):
    """
    Compute CRPS for Gaussian predictive distribution.
    Args:
        mu:      [B, T] predicted mean
        logvar:  [B, T] predicted log-variance
        target:  [B, T] true target values
    Returns:
        crps: scalar (mean CRPS over all points)
    """
    std = (0.5 * logvar).exp()          # [B, T]
    z = (target - mu) / std             # [B, T]

    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))        # PDF φ(z)
    Phi = normal.cdf(z)                        # CDF Φ(z)

    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()


# ──────────────────────────────────────────────────────────────────────────────
# >>>>>>> Your provided flexible data pipeline (UNCHANGED) <<<<<<<
# ──────────────────────────────────────────────────────────────────────────────
def process_seq2seq_data(
        feature_dict,
        *,
        train_ratio        = 0.7,
        norm_features      = ('load', 'temp'),   # ignored: we now normalize ALL features
        output_len         = 24,          # steps each decoder step predicts
        encoder_len_weeks  = 1,
        decoder_len_weeks  = 1,
        num_in_week        = 168,
        device             = None):
    """
    feature_dict: {'load': np.ndarray [weeks, 168], 'featureX': same shape, ...}
    External features are all keys except 'load'.

    NOTE: This version normalizes *all* features with MinMaxScaler (per-feature, global),
    regardless of their names. The 'norm_features' parameter is ignored for compatibility.
    """
    # 1) flatten + scale (normalize ALL features)
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

    # 2) build samples (stride = 1 week)
    X_enc_l, X_dec_in_l, Y_dec_target = [], [], []
    X_enc_ext, X_dec_in_ext = [], []

    last_start = n_weeks - need_weeks
    for w in range(last_start + 1):
        enc_start =  w * num_in_week
        enc_end   = (w + encoder_len_weeks) * num_in_week
        dec_start =  enc_end
        dec_end   =  dec_start + dec_seq_len

        # load sequences
        enc_l = processed['load'][enc_start:enc_end]
        dec_full_l = processed['load'][dec_start:dec_end]

        # encoder/decoder externals
        if K_ext > 0:
            enc_ext = np.stack([processed[k][enc_start:enc_end] for k in ext_keys], axis=-1)  # [enc_len, K]
            dec_ext = np.stack([processed[k][dec_start: dec_start + L] for k in ext_keys], axis=-1)  # [L, K]
        else:
            enc_ext = np.empty((enc_seq_len, 0), dtype=np.float32)
            dec_ext = np.empty((L, 0), dtype=np.float32)

        # targets (sliding windows of length output_len across decoder horizon)
        targets = np.stack([dec_full_l[i:i+output_len] for i in range(L+1)], axis=0)  # [L+1, output_len]

        X_enc_l.append(enc_l)
        X_dec_in_l.append(dec_full_l[:L])
        X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext)
        Y_dec_target.append(targets)

    # 3) pack → tensors
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

    # 4) split
    B = data_tensors['X_enc_l'].shape[0]
    split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}
    test_dict  = {k: v[split:] for k, v in data_tensors.items()}
    return train_dict, test_dict, scalers


def make_loader(split_dict, batch_size, shuffle):
    ds = TensorDataset(
        split_dict['X_enc_l'],
        split_dict['X_enc_ext'],
        split_dict['X_dec_in_l'],
        split_dict['X_dec_in_ext'],
        split_dict['Y_dec_target'],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------
# Data chooser (UNCHANGED)
# ---------------------------
def build_feature_dict(data_name: str):
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
        times, load, temp, workday, season = get_data_oncor_load_weekly()
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
# i-Transformer wrapper (flexible externals)
# ──────────────────────────────────────────────────────────────────────────────
class WeekForecastProbITransformer(nn.Module):
    """
    iTransformer core + a parallel projector for log-variance.
    Returns (mu, logvar) with shape [B, forecast_len].
    Treats each variable (load + externals) as a token along the feature axis.
    The first channel is assumed to be the target 'load'.
    """
    def __init__(self,
                 seq_len: int,             # encoder window length (e.g., 336 for 2 weeks hourly)
                 forecast_len: int = 168,  # one week horizon
                 d_model: int = 256,
                 n_heads: int = 8,
                 e_layers: int = 2,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 factor: int = 3,
                 activation: str = "gelu",
                 use_norm: bool = True):
        super().__init__()

        # Build a tiny config object for official iTransformer Model()
        class C: pass
        cfg = C()
        cfg.seq_len = seq_len
        cfg.pred_len = forecast_len
        cfg.output_attention = False
        cfg.use_norm = use_norm
        cfg.d_model = d_model
        cfg.embed = 'timeF'   # matches the THUML implementation
        cfg.freq  = 'h'       # hourly markers; we’ll pass zeros if you don’t use time marks
        cfg.dropout = dropout
        cfg.class_strategy = 'projection'
        cfg.e_layers = e_layers
        cfg.factor = factor
        cfg.n_heads = n_heads
        cfg.d_ff = d_ff
        cfg.activation = activation

        # The official iTransformer class
        self.core = Model(cfg)

        # Extra projector for log-variance (same E→S map as the mean head)
        self.projector_var = nn.Linear(d_model, forecast_len, bias=True)
        nn.init.constant_(self.projector_var.bias, -3.0)  # stable start (σ≈exp(-1.5))

        self.use_norm = use_norm
        self.pred_len = forecast_len

    def _time_marks(self, B: int, L: int, device: torch.device):
        # Minimal time marks; positional/time enc in iTransformer still provides ordering.
        return torch.zeros(B, L, 4, device=device)

    def forward(self, enc_l, enc_ext=None):
        """
        Inputs:
            enc_l:   [B, L, 1]
            enc_ext: [B, L, K_ext]  (K_ext can be 0)
        Output:
            mu, logvar each [B, pred_len] for the target (channel 0 = 'load')
        """
        # Treat each variate as a token → [B, L, N]
        x_enc = enc_l if (enc_ext is None or enc_ext.numel() == 0) else torch.cat([enc_l, enc_ext], dim=-1)
        B, L, N = x_enc.shape
        device = x_enc.device

        # Time markers placeholder (month/day/weekday/hour) for freq='h'
        x_mark_enc = self._time_marks(B, L, device)

        # === Same normalization as official iTransformer ===
        if self.core.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()  # [B,1,N]
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B,1,N]
            x = x / stdev
        else:
            x, means, stdev = x_enc, None, None

        # Embed (inverted) and encode across variables
        enc_out = self.core.enc_embedding(x, x_mark_enc)          # [B, N', E]
        enc_out, _ = self.core.encoder(enc_out, attn_mask=None)   # [B, N', E]

        # Project to horizon for all tokens, then keep first N (original variates)
        mu_all     = self.core.projector(enc_out).permute(0, 2, 1)[:, :, :N]         # [B, S, N]
        logvar_all = self.projector_var(enc_out).permute(0, 2, 1)[:, :, :N]          # [B, S, N]

        # De-normalize (match the math for mean and variance)
        if self.core.use_norm:
            # μ_denorm = μ_norm * stdev + mean
            mu_all = mu_all * stdev.squeeze(1).unsqueeze(1) + means.squeeze(1).unsqueeze(1)
            # σ^2_denorm = σ^2_norm * stdev^2 ⇒ logσ^2_denorm = logσ^2_norm + 2*log(stdev)
            log_stdev = torch.log(stdev.squeeze(1).clamp_min(1e-6))                  # [B, N]
            logvar_all = logvar_all + 2.0 * log_stdev.unsqueeze(1)                   # [B, S, N]

        # Take the target variable (load) = index 0
        mu     = mu_all[..., 0]       # [B, S]
        logvar = logvar_all[..., 0]   # [B, S]
        return mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (adapted to new batch order)
# ──────────────────────────────────────────────────────────────────────────────
def train_prob_model(model, train_loader, *,
                    epochs: int, lr: float,
                    device: torch.device,
                    save_path: str = "Building_iTransformer_best_model.pt"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best  = float("inf")

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            # new tuple: enc_l, enc_ext, dec_l, dec_ext, tgt
            enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

            # build 168-length ground-truth load vector
            wk_tgt = torch.stack([reconstruct_sequence(t.squeeze(-1)) for t in tgt])  # [B, 168]

            mu, logvar = model(enc_l, enc_ext)
            loss = gaussian_nll_loss(mu, logvar, wk_tgt)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item() * enc_l.size(0)

        avg = running / len(train_loader.dataset)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best @epoch {ep}  NLL {best:.6f}")
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | Train NLL {avg:.6f} | Best {best:.6f}")

    print(f"\n🏁 Training finished. Best epoch NLL = {best:.6f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (adapted to new batch order)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_itransformer_model(model,
                                test_loader,
                                device,
                                model_path="Building_iTransformer_best_model.pt",
                                visualize=True,
                                n_vis_samples=5,
                                global_fig="mean_traj_var_iTransformer.png",
                                quantiles=(0.1, 0.5, 0.9),
                                alpha=0.1   # for 90% prediction interval
                                ):
    """
    Evaluates MSE, NLL, CRPS, plus Quantile Loss (avg over `quantiles`)
    and Winkler Score (for two-sided (1-alpha) interval).
    """
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mse_fn = torch.nn.MSELoss(reduction="mean")
    running_mse = running_nll = running_crps = 0.0
    running_qpin = running_wink = 0.0

    os.makedirs("result", exist_ok=True)

    for batch in test_loader:
        # new tuple: enc_l, enc_ext, dec_l, dec_ext, tgt
        enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

        # ---- forward ------------------------------------------------------
        mu, logvar = model(enc_l, enc_ext)     # [B, H]
        B, H = mu.shape
        sigma = logvar.exp().sqrt()            # [B, H]

        # build ground truth [B, H]
        wk_tgt = torch.stack([reconstruct_sequence(t.squeeze(-1)) for t in tgt])

        # ---- metrics: MSE/NLL/CRPS ---------------------------------------
        running_mse += mse_fn(mu, wk_tgt).item() * B

        nll = 0.5 * (
            logvar
            + torch.log(torch.tensor(2 * np.pi, device=logvar.device))
            + (wk_tgt - mu).pow(2) / logvar.exp()
        )
        running_nll += nll.sum().item()

        crps = crps_gaussian(mu, logvar, wk_tgt)
        running_crps += crps.item() * B

        # ---- Quantile Loss (Pinball) -------------------------------------
        q_losses = []
        for q in quantiles:
            zq = gaussian_icdf(q, device=mu.device)  # scalar
            yq = mu + sigma * zq
            ql = pinball_loss(wk_tgt, yq, q).mean()
            q_losses.append(ql)
        qpin_mean = torch.stack(q_losses).mean()
        running_qpin += qpin_mean.item() * B

        # ---- Winkler Score -----------------------------------------------
        z = gaussian_icdf(1.0 - alpha/2.0, device=mu.device)
        L = mu - z * sigma
        U = mu + z * sigma
        ws = winkler_score(wk_tgt, L, U, alpha).mean()
        running_wink += ws.item() * B

        # ---- visualisation -----------------------------------------------
        if visualize:
            x_axis = np.arange(H)
            for i in range(min(n_vis_samples, B)):
                std_pred = sigma[i].cpu()
                plt.figure(figsize=(4, 2))
                plt.plot(wk_tgt[i].cpu()[:H-2], '--', color='red', label='True')
                plt.plot(mu[i].cpu()[:H-2], color='blue', alpha=0.6, label='Mean Pred')
                plt.fill_between(x_axis[:H-2],
                                 (mu[i].cpu() - std_pred)[:H-2],
                                 (mu[i].cpu() + std_pred)[:H-2],
                                 color='blue', alpha=0.1, label='±1 σ')
                plt.tight_layout()
                plt.ylim(0, 1)
                plt.yticks([0, 0.5, 1], fontsize=14)
                plt.xticks(fontsize=14)
                plt.savefig(f"./result/{data_name}_{model_name}_sample_{i}.pdf")

            plt.figure(figsize=(12, 6))
            for i in range(B):
                std_pred = sigma[i].cpu()
                plt.plot(wk_tgt[i].cpu(), '--', color='grey', linewidth=0.8, alpha=0.4)
                plt.plot(mu[i].cpu(), linewidth=2.0, color='blue',
                         label='Mean Pred' if i == 0 else None)
                plt.fill_between(x_axis, mu[i].cpu() - std_pred, mu[i].cpu() + std_pred,
                                 alpha=0.2, color='red')
            plt.xlabel("Time step");  plt.ylabel("Load")
            plt.title("All Forecasts • Mean + Predicted Variance")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f"./result/{global_fig}", dpi=300)
            visualize = False

    # ---- final metrics ----------------------------------------------------
    num_pts  = len(test_loader.dataset) * H
    test_mse   = running_mse  / len(test_loader.dataset)
    test_nll   = running_nll  / num_pts
    test_crps  = running_crps / len(test_loader.dataset)
    test_qpin  = running_qpin / len(test_loader.dataset)
    test_wink  = running_wink / len(test_loader.dataset)

    print(f"\nTest MSE         : {test_mse:.6f}")
    # print(f"Test NLL         : {test_nll:.6f}")
    print(f"Test CRPS        : {test_crps:.6f}")
    print(f"Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
    print(f"Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")

    return test_mse, test_nll, test_crps, test_qpin, test_wink


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(42)

    # hyper-params ------------------------------------------------------------
    batch_size   = 32
    epochs       = 300
    lr           = 1e-3
    output_len         = 3           # must be < 168 so L > 0
    encoder_len_weeks  = 2           # two-week history
    decoder_len_weeks  = 1           # one-week forecast

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name   = f"iTransformer_v1_{output_len}hours"
    data_name = "ETTH2"  # Spanish Building Consumption Residential Solar Oncor_load ETTH1  ETTH2  GEFCom2014, flores

    model_path   = f"{data_name}_{model_name}_best_model.pt"
    print(f"Using device: {device}")

    # (A) Load & prepare data (flexible externals) ---------------------------
    feature_dict = build_feature_dict(data_name)


    train_data, test_data, _ = process_seq2seq_data(
        feature_dict       = feature_dict,
        train_ratio        = 0.7,
        output_len         = output_len,
        encoder_len_weeks  = encoder_len_weeks,
        decoder_len_weeks  = decoder_len_weeks,
        device             = device)

    n_externals = train_data['X_enc_ext'].shape[-1]
    print(f"K_ext (number of external features) = {n_externals}")

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader  = make_loader(test_data,  batch_size, shuffle=False)

    # Determine encoder seq_len directly from a batch
    sample_batch = next(iter(train_loader))
    enc_len = sample_batch[0].shape[1]  # enc_l has index 0

    # Build model
    model = WeekForecastProbITransformer(
        seq_len=enc_len,  # e.g., 336 for 2 weeks hourly
        forecast_len=168,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=512,
        dropout=0.1,
        use_norm=True
    ).to(device)

    # Train if no checkpoint
    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_prob_model(model, train_loader, epochs=epochs, lr=lr,
                         device=device, save_path=model_path)

    # fresh instance for a clean eval checkpoint
    model_eval = WeekForecastProbITransformer(
        seq_len=enc_len,
        forecast_len=168,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=512,
        dropout=0.1,
        use_norm=True
    ).to(device)

    t1 = time.time()
    evaluate_itransformer_model(model_eval,
                                test_loader,
                                device,
                                model_path=model_path,
                                visualize=True,
                                n_vis_samples=5,
                                global_fig="mean_traj_var_iTransformer.png")
    t2 = time.time()
    print("Eval wall time (s):", t2 - t1)
    # plt.show()

