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

# ──────────────────────────────────────────────────────────────────────────────
# Probabilistic LSTM: 2-week encoder → 1-week load forecast (μ, σ²)
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Probabilistic GRU / RNN: 2-week encoder → 1-week load forecast (μ, σ²)
# ──────────────────────────────────────────────────────────────────────────────
class WeekForecastProbRNN(nn.Module):
    """
    Same I/O contract as WeekForecastProbLSTM but uses a GRU (or RNN) cell.
    Just drop this class in, then instantiate it in main.
    """
    def __init__(self,
                 input_features: int = 4,
                 hidden_size:    int = 128,
                 num_layers:     int = 2,
                 forecast_len:   int = 168,
                 dropout:        float = 0.1,
                 cell_type:      str = "GRU"):   # "GRU" or "RNN"
        super().__init__()

        RNN = nn.GRU if cell_type.upper() == "GRU" else nn.RNN
        self.rnn = RNN(input_features, hidden_size,
                       num_layers, batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0)

        # Linear head outputs both μ and log σ²  → 2×forecast_len
        self.head = nn.Linear(hidden_size, 2 * forecast_len)
        nn.init.constant_(self.head.bias[forecast_len:], -3.0)  # mild log-var bias

    def forward(self, enc_l, enc_t, enc_w, enc_s):
        x = torch.cat([enc_l, enc_t, enc_w, enc_s], dim=-1)      # [B, Tenc, 4]
        _, h_n = self.rnn(x)                                     # h_n: [num_layers, B, hidden]
        h_last = h_n[-1]                                         # [B, hidden]
        out = self.head(h_last)                                  # [B, 336]
        mu, logvar = out.chunk(2, dim=-1)                        # each [B, 168]
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



@torch.no_grad()
def evaluate_prob_lstm_compare(model,
                               test_loader,
                               device,
                               model_path      = "best_prob_rnn.pt",
                               visualize       = True,
                               n_samples       = 20,
                               result_dir      = "result",
                               fig_name        = "mean_traj_var_band.png"):
    """
    Evaluation that mimics your VAE-style visualisation for fair comparison.
    Works for models returning (mu, logvar) per forward pass.
    """
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mse = torch.nn.MSELoss()
    running_mse = 0.0
    running_nll = 0.0
    all_preds, all_targets = [], []

    os.makedirs(result_dir, exist_ok=True)   # ensure folder exists

    for batch in test_loader:
        (enc_l, enc_t, enc_w, enc_s,
         _,      _,      _,      _,      tgt) = [t.to(device) for t in batch]

        B = enc_l.size(0)

        # ------------------------ forward once to get mu, σ ------------------
        mu, logvar = model(enc_l, enc_t, enc_w, enc_s)          # [B, 168] each
        std = logvar.mul(0.5).exp()                             # [B, 168]

        # build ground-truth vector
        wk_tgt = torch.stack([
            reconstruct_sequence(t.squeeze(-1)) for t in tgt
        ])                                                      # [B, 168]

        # ------------------------ Monte-Carlo sampling -----------------------
        samples = []
        for _ in range(n_samples):
            eps    = torch.randn_like(std)                      # [B, 168]
            sample = mu + eps * std                             # [B, 168]
            samples.append(sample.cpu())
        samples = torch.stack(samples, 0)                       # [n_samples, B, 168]

        # empirical mean / std across samples (≙ latent-z runs)
        mean_emp = samples.mean(0)                              # [B, 168]
        std_emp  = samples.std (0)                              # [B, 168]

        # predicted aleatoric std is `std` itself
        all_preds  .extend(mean_emp)
        all_targets.extend(wk_tgt.cpu())
        running_mse += mse(mean_emp, wk_tgt.cpu()).item() * B
        running_nll += gaussian_nll_loss(mu, logvar, wk_tgt).item() * B

        # --------------------------------------------------------------------
        # Visualisation
        # --------------------------------------------------------------------
        if visualize:
            # (1) first five individual comparisons
            for i in range(min(5, B)):
                plt.figure(figsize=(8, 2))
                plt.plot(wk_tgt[i].cpu(), '--', label='True')
                plt.plot(mean_emp[i],       label='Mean pred', alpha=0.9)
                plt.fill_between(np.arange(168),
                                 mean_emp[i]-std_emp[i],
                                 mean_emp[i]+std_emp[i],
                                 color='blue', alpha=0.3, label='±1 std (empirical)')
                # plt.fill_between(np.arange(168),
                #                  mean_emp[i]-std[i].cpu(),
                #                  mean_emp[i]+std[i].cpu(),
                #                  color='red',  alpha=0.2, label='±1 std (predicted)')
                plt.title(f"Mean + Uncertainty (sample {i})")
                # plt.legend();
                plt.savefig(f"./result/{data_name}_{model_name}_sample_{i}.pdf")
                plt.tight_layout()
                plt.show()

            # (2) ALL trajectories + variance band  (saved to disk)
            plt.figure(figsize=(12, 6))
            for i in range(B):
                plt.plot(wk_tgt[i].cpu(), color='gray', linestyle='--',
                         linewidth=0.8, alpha=0.5)
                plt.plot(mean_emp[i], linewidth=2.0,
                         label='Mean pred' if i == 0 else None)
                plt.fill_between(np.arange(168),
                                 mean_emp[i]-std[i].cpu(),
                                 mean_emp[i]+std[i].cpu(),
                                 alpha=0.2, color='red')
            plt.title("Forecasts per Input: Mean + Predicted Variance (All Trajectories)")
            plt.xlabel("Time step")
            plt.ylabel("Forecasted value")
            # plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f"./result/{data_name}_{model_name}_all.pdf")
            plt.show()

    test_mse = running_mse / len(test_loader.dataset)
    test_nll = running_nll / len(test_loader.dataset)
    print(f"\n🧪 Test MSE : {test_mse:.6f}")
    print(f"📉 Test NLL : {test_nll:.6f}")
    return test_mse



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


# ─────────────────────────────────────────────────────────────────────────────
# New-style evaluation for WeekForecastProbRNN / GRU
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_rnn_model(model,
                       test_loader,
                       device,
                       model_path="Building_RNN_best_model.pt",
                       visualize=True,
                       n_vis_samples=5,          # single-series plots
                       global_fig="mean_traj_var_RNN.png"):
    """
    Keeps the look-and-feel of your newer evaluate_model routine but is
    compatible with WeekForecastProbRNN, whose forward() returns (mu, logvar).
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
        mu, logvar = model(enc_l, enc_t, enc_w, enc_s)          # [B, 168]
        B          = mu.size(0)

        # build 168-step ground-truth vector
        wk_tgt = torch.stack(
            [reconstruct_sequence(t.squeeze(-1)) for t in tgt]
        )                                                       # [B, 168]

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
            horizon = mu.size(1)
            x_axis  = np.arange(horizon)

            # (1) up to n_vis_samples individual traces
            for i in range(min(n_vis_samples, B)):
                std_pred = logvar[i].exp().sqrt().cpu()

                plt.figure(figsize=(4, 2))
                print(wk_tgt.shape)
                plt.plot(wk_tgt[i].cpu()[:166], '--', color='red',  label='True')
                plt.plot(mu[i].cpu()[:166],   color='blue',  alpha=0.6, label='Mean Pred')
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
            visualize = False

    # ── final metrics ───────────────────────────────────────────────────────
    num_pts  = len(test_loader.dataset) * mu.size(1)   # batches × 168
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
    model_name = "RNN"
    data_name = "Solar"  # Spanish Consumption Residential Solar
    model_path = f"{data_name}_{model_name}_best_model.pt"

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
        output_len         = 3,          # must stay <168 so L>0
        encoder_len_weeks  = 2,           # two-week history
        decoder_len_weeks  = 1,           # one week to predict
        device             = device)

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader  = make_loader(test_data,  batch_size, shuffle=False)

    # (2) Build, train, evaluate ---------------------------------------------
    model = WeekForecastProbRNN(  # ← changed name!
        input_features=4,
        hidden_size=128,
        num_layers=2,
        forecast_len=168,
        cell_type="GRU"  # or "RNN" for tanh cells
    ).to(device)

    import os
    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_prob_model(model, train_loader, epochs=epochs, lr=lr,  device=device, save_path=model_path)

    # fresh instance for a clean eval checkpoint
    model = WeekForecastProbRNN(  # ← changed name!
        input_features=4,
        hidden_size=128,
        num_layers=2,
        forecast_len=168,
        cell_type="GRU"  # or "RNN" for tanh cells
    ).to(device)


    evaluate_rnn_model(model,
                       test_loader,
                       device,
                       model_path=model_path,
                       visualize=True,
                       n_vis_samples=5,
                       global_fig="mean_traj_var_RNN.png")
