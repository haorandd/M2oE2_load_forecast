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
from torch.distributions.normal import Normal
import math


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Helper: torch-native Gaussian NLL
# ──────────────────────────────────────────────────────────────────────────────
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

class WeekForecastProbCNNGRU(nn.Module):
    """
    1) Concatenate the four covariates → [B, Tenc, 4]
    2) Two causal-ish 1-D conv layers pick up local hourly/diurnal patterns
    3) GRU summarises the full 2-week context
    4) Head outputs μ and log σ² for 168-h forecast
    """
    def __init__(self,
                 input_features: int = 4,
                 conv_channels:  tuple = (32, 64),
                 kernel_size:    int   = 3,
                 gru_hidden:     int   = 128,
                 num_layers:     int   = 1,
                 forecast_len:   int   = 168,
                 dropout:        float = 0.1):
        super().__init__()

        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, c1, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(c2, gru_hidden,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        # head delivers both μ and log σ² (aleatoric) → 2×forecast_len
        self.head = nn.Linear(gru_hidden, 2 * forecast_len)
        nn.init.constant_(self.head.bias[forecast_len:], -3.0)  # gentle log-var bias

    def forward(self, enc_l, enc_t, enc_w, enc_s):
        # concat features → [B, Tenc, 4] → [B, 4, Tenc]
        x = torch.cat([enc_l, enc_t, enc_w, enc_s], dim=-1).permute(0, 2, 1)
        x = self.conv(x)                      # [B, c2, Tenc]
        x = x.permute(0, 2, 1)                # [B, Tenc, c2]

        _, h_n = self.gru(x)                  # h_n: [num_layers, B, gru_hidden]
        h_last = h_n[-1]                      # [B, gru_hidden]

        out = self.head(h_last)               # [B, 336]
        mu, logvar = out.chunk(2, dim=-1)     # each [B, 168]
        return mu, logvar

# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_prob_model(model, train_loader, *,
                    epochs: int, lr: float,
                    device: torch.device,
                    save_path: str = "Building_LSTM_best_model.pt"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best  = float("inf")

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            (enc_l, enc_t, enc_w, enc_s,
             _,      _,      _,      _,      tgt) = [t.to(device) for t in batch]

            # build 168-length ground-truth load vector
            wk_tgt = torch.stack([
                reconstruct_sequence(t.squeeze(-1))           # [168]
                for t in tgt
            ])                                                # [B, 168]

            mu, logvar = model(enc_l, enc_t, enc_w, enc_s)
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


# ─────────────────────────────────────────────────────────────────────────────
# New-style evaluation for WeekForecastProbCNNGRU
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_cnn_gru_model(model,
                           test_loader,
                           device,
                           model_path="Building_CNNGRU_best_model.pt",
                           visualize=True,
                           n_vis_samples=5,
                           global_fig="mean_traj_var_CNNGRU.png"):
    """
    Modern visual/metric routine for WeekForecastProbCNNGRU.
    The model’s forward() returns (mu, logvar) with shape [B, 168].
    """
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mse_fn   = torch.nn.MSELoss(reduction="mean")
    running_mse, running_nll, running_crps = 0.0, 0.0, 0.0

    os.makedirs("result", exist_ok=True)

    for batch in test_loader:
        (enc_l, enc_t, enc_w, enc_s,
         _,      _,      _,      _,      tgt) = [t.to(device) for t in batch]

        # ── forward ──────────────────────────────────────────────────────────
        mu, logvar = model(enc_l, enc_t, enc_w, enc_s)           # [B, 168]
        B, horizon = mu.size()

        # build ground truth
        wk_tgt = torch.stack(
            [reconstruct_sequence(t.squeeze(-1)) for t in tgt]
        )                                                        # [B, 168]

        # accumulate metrics
        running_mse += mse_fn(mu, wk_tgt).item() * B
        nll = 0.5 * (
            logvar
            + torch.log(torch.tensor(2 * np.pi, device=logvar.device))
            + (wk_tgt - mu).pow(2) / logvar.exp()
        )
        running_nll += nll.sum().item()


        crps = crps_gaussian(mu, logvar, wk_tgt)
        running_crps += crps.item() * B

        # ── visualisation ───────────────────────────────────────────────────
        if visualize:
            x_axis = np.arange(horizon)

            # (1) individual traces
            for i in range(min(n_vis_samples, B)):
                std_pred = logvar[i].exp().sqrt().cpu()

                plt.figure(figsize=(4, 2))
                plt.plot(wk_tgt[i].cpu()[:166], '--', color='red',  label='True')
                plt.plot(mu[i].cpu()[:166],      color='blue',     alpha=0.6,
                         label='Mean Pred')
                plt.fill_between(x_axis[:166],
                                 (mu[i].cpu() - std_pred)[:166],
                                 (mu[i].cpu() + std_pred)[:166],
                                 color='blue', alpha=0.1,
                                 label='±1 σ (pred.)')
                plt.tight_layout()
                plt.ylim(0, 1)
                plt.yticks([0, 0.5, 1], fontsize = 14)
                plt.xticks(fontsize=14)
                plt.savefig(f"./result/{data_name}_{model_name}_sample_{i}.pdf")
                plt.show()

            # (2) global overlay
            plt.figure(figsize=(12, 6))
            for i in range(B):
                std_pred = logvar[i].exp().sqrt().cpu()
                plt.plot(wk_tgt[i].cpu(), '--', color='grey',
                         linewidth=0.8, alpha=0.4)
                plt.plot(mu[i].cpu(), linewidth=2.0, color='blue',
                         label='Mean Pred' if i == 0 else None)
                plt.fill_between(x_axis,
                                 mu[i].cpu() - std_pred,
                                 mu[i].cpu() + std_pred,
                                 alpha=0.2, color='red')
            plt.xlabel("Time step");  plt.ylabel("Load")
            plt.title("All Forecasts • Mean + Predicted Variance")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f"./result/{global_fig}", dpi=300)
            # plt.show()
            visualize = False  #

    # ── final metrics ───────────────────────────────────────────────────────
    num_pts  = len(test_loader.dataset) * horizon
    test_mse = running_mse / len(test_loader.dataset)
    test_nll = running_nll / num_pts

    test_crps = running_crps / len(test_loader.dataset)


    print(f"\n🧪 Test MSE : {test_mse:.6f}")
    # print(f"🧪 Test NLL : {test_nll:.6f}")
    print(f"🧪 Test CRPS: {test_crps:.6f}")

    return test_mse, test_nll, test_crps




if __name__ == "__main__":
    set_seed(42)

    # hyper-params ------------------------------------------------------------
    batch_size   = 32
    epochs       = 300
    lr           = 1e-3
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "CNNGRU"
    data_name = "Solar"  # Spanish  Building Consumption Residential Solar
    model_path = f"{data_name}_{model_name}_best_model.pt"
    print(f"Using device: {device}")

    # (A) Load & prepare data ------------------------------------------------
    if data_name == "Building":
        times, load, temp, workday, season = get_data_building_weather_weekly()
    elif data_name == "Spanish":
        times, load, temp, workday, season = get_data_spanish_weekly()
    elif data_name == "Consumption":
        times, load, temp, workday, season = get_data_power_consumption_weekly()
    elif data_name == "Residential":
        times, load, temp, workday, season = get_data_residential_weekly()
    elif data_name == "Solar":
        times, load, temp, workday, season= get_data_solar_weather_weekly()

    feature_dict = dict(load=load, temp=temp, workday=workday, season=season)


    train_data, test_data, _ = process_seq2seq_data(
        feature_dict       = feature_dict,
        train_ratio        = 0.7,
        output_len         = 3,          # 24 must stay <168 so L>0
        encoder_len_weeks  = 2,           # two-week history
        decoder_len_weeks  = 1,           # one week to predict
        device             = device)

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader  = make_loader(test_data,  batch_size, shuffle=False)

    model = WeekForecastProbCNNGRU(
        input_features=4,
        conv_channels=(32, 64),  # feel free to tune
        kernel_size=3,
        gru_hidden=128,
        num_layers=1,
        forecast_len=168,
        dropout=0.1
    ).to(device)


    import os
    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_prob_model(model, train_loader,  epochs=epochs, lr=lr,  device=device, save_path=model_path)

    model = WeekForecastProbCNNGRU(
        input_features=4,
        conv_channels=(32, 64),  # feel free to tune
        kernel_size=3,
        gru_hidden=128,
        num_layers=1,
        forecast_len=168,
        dropout=0.1
    ).to(device)


    evaluate_cnn_gru_model(model,
                           test_loader,
                           device,
                           model_path=model_path,
                           visualize=True,
                           n_vis_samples=5,
                           global_fig="mean_traj_var_CNNGRU.png")