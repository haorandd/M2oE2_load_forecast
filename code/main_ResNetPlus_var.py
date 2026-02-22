import os, math, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from data_utils import *
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    std = (0.5 * logvar).exp()  # [B, T]
    z = (target - mu) / std  # [B, T]

    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))  # PDF φ(z)
    Phi = normal.cdf(z)  # CDF Φ(z)

    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()


def gaussian_icdf(p, device):
    # Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        2 * torch.as_tensor(p, device=device) - 1)


def pinball_loss(y, yq, q):
    e = y - yq
    return torch.where(e >= 0, q * e, (q - 1) * e)  # elementwise; lower is better


def winkler_score(y, L, U, alpha):
    width = (U - L)
    below = (L - y).clamp(min=0.0)
    above = (y - U).clamp(min=0.0)
    return width + (2.0 / alpha) * (below + above)  # elementwise; lower is better


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


def process_seq2seq_data(
        feature_dict,
        *,
        train_ratio=0.7,
        norm_features=('load', 'temp'),  # ignored: we now normalize ALL features
        output_len=24,  # steps each decoder step predicts
        encoder_len_weeks=1,
        decoder_len_weeks=1,
        num_in_week=168,
        device=None):
    """
    feature_dict: {'load': np.ndarray [weeks, 168], 'featureX': same shape, ...}
    External features are all keys except 'load'.
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
        enc_start = w * num_in_week
        enc_end = (w + encoder_len_weeks) * num_in_week
        dec_start = enc_end
        dec_end = dec_start + dec_seq_len

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
        targets = np.stack([dec_full_l[i:i + output_len] for i in range(L + 1)], axis=0)  # [L+1, output_len]

        X_enc_l.append(enc_l)
        X_dec_in_l.append(dec_full_l[:L])
        X_enc_ext.append(enc_ext)
        X_dec_in_ext.append(dec_ext)
        Y_dec_target.append(targets)

    # 3) pack → tensors
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)

    data_tensors = {
        'X_enc_l': to_tensor(np.array(X_enc_l)).unsqueeze(-1),  # [B, enc_len, 1]
        'X_enc_ext': to_tensor(np.array(X_enc_ext)),  # [B, enc_len, K_ext]
        'X_dec_in_l': to_tensor(np.array(X_dec_in_l)).unsqueeze(-1),  # [B, L, 1]
        'X_dec_in_ext': to_tensor(np.array(X_dec_in_ext)),  # [B, L, K_ext]
        'Y_dec_target': to_tensor(np.array(Y_dec_target)).unsqueeze(-1),  # [B, L+1, output_len, 1]
    }

    for k, v in data_tensors.items():
        print(f"{k:15s} {tuple(v.shape)}")

    # 4) split
    B = data_tensors['X_enc_l'].shape[0]
    split = int(train_ratio * B)
    train_dict = {k: v[:split] for k, v in data_tensors.items()}
    test_dict = {k: v[split:] for k, v in data_tensors.items()}
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
        feature_dict = {'HUFL': HUFL, 'HULL': HULL, 'MUFL': MUFL, 'MULL': MULL, 'LUFL': LUFL, 'LULL': LULL, 'load': OT,
                        'workday': workday, 'season': season_feat}
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
# ResNetPlus Model Component (Based on "Short-Term Load Forecasting With Deep Residual Networks")
# ──────────────────────────────────────────────────────────────────────────────
class ResNetPlusBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        # Dense block layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.selu1 = nn.SELU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)

        # SELU activation after adding residual and shortcut
        self.selu2 = nn.SELU()

    def forward(self, x, x_init):
        # x_init represents the dense shortcut connection from earlier in the network
        out = self.fc1(x)
        out = self.selu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)

        # ResNetPlus introduces dense shortcuts: x (immediate skip) + out (residual) + x_init (dense skip)
        return self.selu2(out + x + x_init)


class WeekForecastProbResNetPlus(nn.Module):
    """
    Drop-in replacement for STLF utilizing ResNetPlus architecture.
    Outputs predictive mean and logvar to match existing training/eval frameworks.
    """

    def __init__(self,
                 input_features: int,  # Kept for compatibility mapping
                 hidden_size: int = 256,
                 num_blocks: int = 4,  # Adjust depth (Paper tests up to ~60 layers)
                 forecast_len: int = 168,
                 dropout: float = 0.1):
        super().__init__()
        self.forecast_len = forecast_len

        # Basic structure maps raw input -> latent space using SELU
        self.input_layer = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.SELU(),
            nn.Dropout(dropout)
        )

        # Stacked ResNetPlus Blocks
        self.blocks = nn.ModuleList([
            ResNetPlusBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])

        # Dual-head output for Gaussian Probabilistic Forecasting (mu, logvar)
        self.out_mu = nn.Linear(hidden_size, forecast_len)
        self.out_logvar = nn.Linear(hidden_size, forecast_len)

    def forward(self, enc_l, enc_ext=None):
        """
        enc_l:   [B, Tenc, 1]
        enc_ext: [B, Tenc, K_ext] (K_ext can be 0)
        returns:
          mu:     [B, forecast_len]
          logvar: [B, forecast_len]
        """
        # Combine load and external features
        x = enc_l if (enc_ext is None or enc_ext.numel() == 0) else torch.cat([enc_l, enc_ext], dim=-1)
        # Flatten all encoder timesteps/features into a single vector per sample
        x = x.reshape(x.size(0), -1)  # [B, Tenc*(1+K)]

        # Initial representation
        x_init = self.input_layer(x)
        h = x_init

        # Pass through ResNetPlus blocks
        for block in self.blocks:
            h = block(h, x_init)

        # Final Point + Probabilistic Forecast mapping
        mu = self.out_mu(h)
        logvar = self.out_logvar(h)

        # Optional: clamp logvar to avoid numerical instability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_prob_lstm(model, train_loader, *,
                    epochs: int, lr: float,
                    device: torch.device,
                    save_path: str = "_model_best.pt"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train();
        running = 0.0
        for batch in train_loader:
            # new tuple: enc_l, enc_ext, dec_l, dec_ext, tgt
            enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

            # build 168-length ground-truth load vector
            wk_tgt = torch.stack([
                reconstruct_sequence(t.squeeze(-1))  # [168]
                for t in tgt
            ])  # [B, 168]

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
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_lstm_model(model,
                        test_loader,
                        device,
                        model_path="_model_best.pt",
                        visualize=True,
                        n_vis_samples=5,
                        global_fig="mean_traj_var.png",
                        quantiles=(0.1, 0.5, 0.9),  # for pinball loss
                        alpha=0.1  # Winkler for (1-alpha) PI
                        ):
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mse_fn = torch.nn.MSELoss(reduction="mean")
    running_mse, running_nll, running_crps = 0.0, 0.0, 0.0
    running_qpin, running_wink = 0.0, 0.0

    os.makedirs("result", exist_ok=True)

    for batch in test_loader:
        # new tuple: enc_l, enc_ext, dec_l, dec_ext, tgt
        enc_l, enc_ext, _, _, tgt = [t.to(device) for t in batch]

        # ── forward ──────────────────────────────────────────────────────────
        mu, logvar = model(enc_l, enc_ext)  # [B, H]
        B, horizon = mu.size()
        sigma = logvar.exp().sqrt()  # [B, H]

        # build ground truth [B, H]
        wk_tgt = torch.stack(
            [reconstruct_sequence(t.squeeze(-1)) for t in tgt]
        )

        # ── core metrics ────────────────────────────────────────────────────
        running_mse += mse_fn(mu, wk_tgt).item() * B
        nll = 0.5 * (logvar + torch.log(torch.tensor(2 * np.pi, device=logvar.device))
                     + (wk_tgt - mu).pow(2) / logvar.exp())
        running_nll += nll.sum().item()

        crps = crps_gaussian(mu, logvar, wk_tgt)  # mean over batch
        running_crps += crps.item() * B

        # ── Quantile (pinball) loss ----------------------------------------
        q_losses = []
        for q in quantiles:
            zq = gaussian_icdf(q, device=mu.device)  # scalar
            yq = mu + sigma * zq  # predicted q-quantile
            ql = pinball_loss(wk_tgt, yq, q).mean()  # avg over B*H
            q_losses.append(ql)
        qpin_mean = torch.stack(q_losses).mean()
        running_qpin += qpin_mean.item() * B

        # ── Winkler score for (1-alpha) PI ---------------------------------
        z = gaussian_icdf(1.0 - alpha / 2.0, device=mu.device)
        L = mu - z * sigma
        U = mu + z * sigma
        ws = winkler_score(wk_tgt, L, U, alpha).mean()  # avg over B*H
        running_wink += ws.item() * B

        if visualize:
            x_axis = np.arange(horizon)

            # (1) individual plots
            for i in range(min(n_vis_samples, B)):
                std_pred = sigma[i].cpu()
                plt.figure(figsize=(4, 2))
                plt.plot(wk_tgt[i].cpu()[:166], '--', color='red', label='True')
                plt.plot(mu[i].cpu()[:166], color='blue', alpha=0.6, label='Mean Pred')
                plt.fill_between(x_axis[:166],
                                 (mu[i].cpu() - std_pred)[:166],
                                 (mu[i].cpu() + std_pred)[:166],
                                 color='blue', alpha=0.1, label='±1 σ (pred.)')
                plt.tight_layout()
                plt.ylim(0, 1)
                plt.yticks([0, 0.5, 1], fontsize=14)
                plt.xticks(fontsize=14)
                plt.savefig(f"./result/{data_name}_{model_name}_sample_{i}.pdf")

            # (2) global overlay
            plt.figure(figsize=(12, 6))
            for i in range(B):
                std_pred = sigma[i].cpu()
                plt.plot(wk_tgt[i].cpu(), '--', color='grey', linewidth=0.8, alpha=0.4)
                plt.plot(mu[i].cpu(), linewidth=2.0, color='blue',
                         label='Mean Pred' if i == 0 else None)
                plt.fill_between(x_axis, mu[i].cpu() - std_pred, mu[i].cpu() + std_pred,
                                 alpha=0.2, color='red')
            plt.xlabel("Time step");
            plt.ylabel("Load")
            plt.title("All Forecasts • Mean + Predicted Variance")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f"./result/{global_fig}", dpi=300)
            visualize = False

    # ── final metrics ───────────────────────────────────────────────────────
    num_pts = len(test_loader.dataset) * horizon
    test_mse = running_mse / len(test_loader.dataset)
    test_nll = running_nll / num_pts
    test_crps = running_crps / len(test_loader.dataset)
    test_qpin = running_qpin / len(test_loader.dataset)
    test_wink = running_wink / len(test_loader.dataset)

    print(f"\nTest MSE         : {test_mse:.6f}")
    print(f"Test CRPS        : {test_crps:.6f}")
    print(f"Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
    print(f"Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1 - alpha) * 100:.0f}% PI)")

    return test_mse, test_nll, test_crps, test_qpin, test_wink


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(42)

    # hyper-params ------------------------------------------------------------
    batch_size = 32
    epochs = 300
    lr = 1e-3
    output_len = 3  # must be < 168 so L>0
    encoder_len_weeks = 2  # two-week history
    decoder_len_weeks = 1  # one week to predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Updated Model name for naming exports and paths
    model_name = f"ResNetPlus_{output_len}hours"
    data_name = "ETTH2"  # Spanish Building Consumption Residential Solar Oncor_load ETTH1 ETTH2 GEFCom2014, flores
    model_path = f"{data_name}_{model_name}_best_model.pt"

    # (A) Load & prepare data (flexible externals) ---------------------------
    feature_dict = build_feature_dict(data_name)

    train_data, test_data, _ = process_seq2seq_data(
        feature_dict=feature_dict,
        train_ratio=0.7,
        output_len=output_len,
        encoder_len_weeks=encoder_len_weeks,
        decoder_len_weeks=decoder_len_weeks,
        device=device)

    n_externals = train_data['X_enc_ext'].shape[-1]
    print(f"K_ext (number of external features) = {n_externals}")

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader = make_loader(test_data, batch_size, shuffle=False)

    # (2) Build, train, evaluate ---------------------------------------------
    # Replaced WeekForecastProbQRNN with WeekForecastProbResNetPlus
    model = WeekForecastProbResNetPlus(
        input_features=1 + n_externals,
        hidden_size=256,
        num_blocks=4,
        forecast_len=168,
        dropout=0.1
    ).to(device)

    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_prob_lstm(model, train_loader, epochs=epochs, lr=lr, device=device, save_path=model_path)

    # fresh instance for a clean eval checkpoint
    model_eval = WeekForecastProbResNetPlus(
        input_features=1 + n_externals,
        hidden_size=256,
        num_blocks=4,
        forecast_len=168,
        dropout=0.1
    ).to(device)

    time1 = time.time()
    evaluate_lstm_model(model_eval,
                        test_loader,
                        device,
                        model_path=model_path,
                        visualize=True,
                        n_vis_samples=5,
                        global_fig="mean_traj_var_ResNetPlus.png")
    time2 = time.time()
    print("Eval wall time (s):", time2 - time1)