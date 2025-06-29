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
from Informer2020.models.model import Informer

from torch.distributions.normal import Normal
import math

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
    std = (0.5 * logvar).exp()          # [B, T]
    z = (target - mu) / std             # [B, T]

    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))        # PDF φ(z)
    Phi = normal.cdf(z)                        # CDF Φ(z)

    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()


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
    #
    # nll = 0.5 * (logvar + np.log(2 * np.pi) + ((target - mu) ** 2) / logvar.exp())
    # return nll.mean()  # average over all elements



# ──────────────────────────────────────────────────────────────────────────────
# Probabilistic Informer: 2-week encoder → 1-week load forecast (μ, σ²)
# ──────────────────────────────────────────────────────────────────────────────
class WeekForecastProbInformer(nn.Module):
    """
    A thin wrapper around the original Informer implementation that
    • accepts the four feature tensors already produced by your data pipeline
      (shape [B, Tenc, 1] each);
    • returns (mu, logvar) with shape [B, 168] – exactly like the LSTM version.
    Everything else (training loop, loss, evaluation code, visualisations)
    stays *unchanged*.
    """
    def __init__(self,
                 input_features : int  = 4,      # load, temp, work-day, season
                 forecast_len   : int  = 168,    # hours in one week
                 encoder_weeks  : int  = 2,      # history length used above
                 label_len      : int  = 24,     # “teacher-forcing” context fed to decoder
                 d_model        : int  = 256,
                 n_heads        : int  = 8,
                 e_layers       : int  = 3,
                 d_layers       : int  = 2,
                 d_ff           : int  = 512,
                 dropout        : float = 0.1,
                 device         : torch.device = torch.device("cpu")):
        super().__init__()

        self.seq_len  = encoder_weeks * 7 * 24          # 336 if encoder_weeks = 2
        self.label_len = label_len                      # context passed to decoder
        self.pred_len  = forecast_len                   # 168

        # --------------- core Informer ----------------
        self.informer = Informer(
            enc_in     = input_features,
            dec_in     = input_features,
            c_out      = 2,                 # μ and log σ²   (per step)
            seq_len    = self.seq_len,
            label_len  = self.label_len,
            out_len    = self.pred_len,
            factor     = 5,
            d_model    = d_model,
            n_heads    = n_heads,
            e_layers   = e_layers,
            d_layers   = d_layers,
            d_ff       = d_ff,
            dropout    = dropout,
            attn       = 'prob',
            embed      = 'fixed',           # same as original paper
            freq       = 'h',
            activation = 'gelu',
            device     = device
        )

        # encourage reasonable initial log-variance
        nn.init.constant_(self.informer.projection.bias[self.pred_len:], -3.0)

    # --------------------------------------------------------------------- #
    # helper that builds the “time-feature” tensor Informer expects
    # (hour, day-of-week, etc.).  For our purposes we can get away with
    # zeros – the fixed positional embedding still provides ordering.
    # --------------------------------------------------------------------- #
    def _dummy_time_marks(self, B: int, L: int, device: torch.device):
        return torch.zeros(B, L, 4, device=device)  # 4 is arbitrary >0

    def forward(self, enc_l, enc_t, enc_w, enc_s):
        """
        Inputs:  four tensors with shape [B, Tenc, 1] (Tenc == self.seq_len)
        Returns: mu, logvar  ∈ ℝ^{B×168}
        """
        device = enc_l.device
        B      = enc_l.size(0)

        # --------------------------- build encoder / decoder streams --------
        x_enc = torch.cat([enc_l, enc_t, enc_w, enc_s], dim=-1)          # [B, 336, 4]
        x_mark_enc = self._dummy_time_marks(B, self.seq_len, device)     # [B, 336, 4]

        # decoder input = last `label_len` points from history + zeros padding
        x_dec_hist  = x_enc[:, -self.label_len:, :]                      # [B, 24, 4]
        x_dec_pred  = torch.zeros(B, self.pred_len, x_enc.size(-1),
                                  device=device)                         # [B, 168, 4]
        x_dec       = torch.cat([x_dec_hist, x_dec_pred], dim=1)         # [B, 24+168, 4]
        x_mark_dec  = self._dummy_time_marks(B,
                                             self.label_len + self.pred_len,
                                             device)                     # [B, 192, 4]

        out = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec)        # [B, 168, 2]
        mu, logvar = out[..., 0], out[..., 1]                            # each [B, 168]
        return mu, logvar


def train_prob_lstm(model, train_loader, *,
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
# New-style evaluation for WeekForecastProbInformer
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_informer_model(model,
                            test_loader,
                            device,
                            model_path="Building_Informer_best_model.pt",
                            visualize=True,
                            n_vis_samples=5,
                            global_fig="mean_traj_var_Informer.png"):
    """
    Works with WeekForecastProbInformer whose forward() returns (mu, logvar)
    with shape [B, 168].  Metrics + plots align with your “modern” format.
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

        # ground truth
        wk_tgt = torch.stack(
            [reconstruct_sequence(t.squeeze(-1)) for t in tgt]
        )                                                        # [B, 168]

        # metrics
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
    # test_nll = running_nll / len(test_loader.dataset)
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
    model_name = "Informer"
    data_name = "Residential"  # Spanish Consumption Residential Solar
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
        output_len         = 24,          # must stay <168 so L>0
        encoder_len_weeks  = 2,           # two-week history
        decoder_len_weeks  = 1,           # one week to predict
        device             = device)

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader  = make_loader(test_data,  batch_size, shuffle=False)

    # (2) Build, train, evaluate ---------------------------------------------
    model = WeekForecastProbInformer(
        input_features=4,
        forecast_len=168,
        encoder_weeks=2,  # must match your data-prep
        label_len=24,  # 1-day decoder context; tune if you wish
        d_model=256,  #
        n_heads=8,
        e_layers=3,
        d_layers=2,
        dropout=0.1,
        device=device
    ).to(device)

    import os
    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_prob_lstm(model, train_loader, epochs=epochs, lr=lr, device=device, save_path=model_path)


    model = WeekForecastProbInformer(
        input_features=4,
        forecast_len=168,
        encoder_weeks=2,  # must match your data-prep
        label_len=24,  # 1-day decoder context; tune if you wish
        d_model=256,  # ↓ feel free to tweak
        n_heads=8,
        e_layers=3,
        d_layers=2,
        dropout=0.1,
        device=device
    ).to(device)


    evaluate_informer_model(model,
                            test_loader,
                            device,
                            model_path=model_path,  # e.g. "Building_Informer_best_model.pt"
                            visualize=True,
                            n_vis_samples=5,
                            global_fig="mean_traj_var_Informer.png")