from data_utils import *
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions.normal import Normal

# ----------------------------------------------------------------------------
# Helper: Gaussian NLL (unchanged)
# ----------------------------------------------------------------------------

def gaussian_nll_loss(mu: torch.Tensor, logvar: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    nll = 0.5 * (logvar + math.log(2 * math.pi) +
                 (target - mu).pow(2) / logvar.exp())
    return nll.mean()


# ----------------------------------------------------------------------------
# Probabilistic ARIMA backbone *with* scaling
# ----------------------------------------------------------------------------

class WeekForecastProbARIMA(nn.Module):
    """Differentiable AR(p) with built‑in z‑scaling.

    Parameters
    ----------
    p : int
        AR order (we default to 48 ⇒ two days worth of lags).
    forecast_len : int
        Horizon (168 → one week).
    load_mean, load_std : float
        Dataset statistics used for on‑the‑fly standardisation.
    """

    def __init__(self, p: int, forecast_len: int,
                 load_mean: float, load_std: float):
        super().__init__()
        self.p            = p
        self.forecast_len = forecast_len

        # store scaling factors as buffers so they move with .to(device)
        self.register_buffer("mu",  torch.tensor(load_mean))
        self.register_buffer("sigma", torch.tensor(load_std))

        # AR coefficients, intercept, log‑variance
        self.phi        = nn.Parameter(torch.randn(p) * 0.05)  # small N(0,0.05²)
        self.bias       = nn.Parameter(torch.zeros(1))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, enc_l, enc_t, enc_w, enc_s):
        # ------------------------------------------------------------------
        # 1.  standardise recent load history
        # ------------------------------------------------------------------
        y_hist = enc_l.squeeze(-1)                   # [B, 336]
        y_hist_std = (y_hist - self.mu) / self.sigma

        last_vals = y_hist_std[:, -self.p:]          # [B, p]
        mu_std_preds = []
        for _ in range(self.forecast_len):
            y_pred_std = self.bias + (last_vals * self.phi).sum(dim=1)  # [B]
            mu_std_preds.append(y_pred_std)
            last_vals = torch.cat([last_vals[:, 1:], y_pred_std.unsqueeze(1)], dim=1) if self.p > 1 else y_pred_std.unsqueeze(1)

        mu_std_preds = torch.stack(mu_std_preds, dim=1)  # [B, 168]

        # ------------------------------------------------------------------
        # 2. back‑transform to original scale *including* variance
        # ------------------------------------------------------------------
        mu_orig = mu_std_preds * self.sigma + self.mu
        logvar_orig = self.log_sigma2 + 2 * torch.log(self.sigma)   # var scales by σ²
        logvar_orig = logvar_orig.expand_as(mu_orig)
        return mu_orig, logvar_orig


# ----------------------------------------------------------------------------
# Training loop (minimal tweaks — weight decay + higher LR)
# ----------------------------------------------------------------------------

def train_prob_model(model, train_loader, *, epochs: int, lr: float,
                     device: torch.device, save_path: str = "Building_ARIMA_best_model.pt",
                     weight_decay: float = 1e-4):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best  = float("inf")

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            (enc_l, enc_t, enc_w, enc_s, _, _, _, _, tgt) = [t.to(device) for t in batch]

            wk_tgt = torch.stack([reconstruct_sequence(t.squeeze(-1)) for t in tgt])
            mu, logvar = model(enc_l, enc_t, enc_w, enc_s)
            loss = gaussian_nll_loss(mu, logvar, wk_tgt)

            optim.zero_grad(); loss.backward(); optim.step()
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


# ----------------------------------------------------------------------------
# CRPS helper (unchanged)
# ----------------------------------------------------------------------------

def crps_gaussian(mu, logvar, target):
    std = (0.5 * logvar).exp()
    z   = (target - mu) / std
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z)); Phi = normal.cdf(z)
    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()


# ----------------------------------------------------------------------------
# Evaluation (unchanged logic but uses improved model)
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_arima_model(model, test_loader, device, *,
                         model_path="Building_ARIMA_best_model.pt", visualize=True,
                         n_vis_samples=5, global_fig="mean_traj_var_ARIMA.png"):
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mse_fn = nn.MSELoss(reduction="mean")
    running_mse = running_nll = running_crps = 0.0
    os.makedirs("result", exist_ok=True)

    for batch in test_loader:
        (enc_l, enc_t, enc_w, enc_s, _, _, _, _, tgt) = [t.to(device) for t in batch]

        mu, logvar = model(enc_l, enc_t, enc_w, enc_s)
        B = mu.size(0)
        wk_tgt = torch.stack([reconstruct_sequence(t.squeeze(-1)) for t in tgt])

        running_mse += mse_fn(mu, wk_tgt).item() * B
        nll = 0.5 * (logvar + math.log(2 * math.pi) + (wk_tgt - mu).pow(2) / logvar.exp())
        running_nll += nll.sum().item()
        running_crps += crps_gaussian(mu, logvar, wk_tgt).item() * B

        # ── visualisation ────────────────────────────────────────────────
        if visualize:
            horizon = mu.size(1)
            x_axis = np.arange(horizon)
            mu_np, std_np = mu.detach().cpu(), logvar.detach().cpu().exp().sqrt()
            tgt_np = wk_tgt.detach().cpu()

            # individual traces
            for i in range(min(n_vis_samples, B)):
                print(i, n_vis_samples, B)
                plt.figure(figsize=(4, 2))
                plt.plot(tgt_np[i][:166], "--", color="red", label="True")
                plt.plot(mu_np[i][:166],  color="blue", alpha=0.7, label="Mean Pred")
                plt.fill_between(x_axis[:166], (mu_np[i] - std_np[i])[:166], (mu_np[i] + std_np[i])[:166], alpha=0.12, color="blue")
                plt.tight_layout()
                plt.ylim(0, 1)
                plt.yticks([0, 0.5, 1], fontsize = 14)
                plt.xticks(fontsize=14)
                plt.savefig(f"./result/{data_name}_{model_name}_sample_{i}.pdf")
                plt.show()

            # global overlay
            plt.figure(figsize=(12, 6))
            for i in range(B):
                plt.plot(tgt_np[i], "--", color="grey", linewidth=0.8, alpha=0.3)
                plt.plot(mu_np[i], linewidth=1.6, color="blue", alpha=0.8,
                         label="Mean Pred" if i == 0 else None)
                plt.fill_between(x_axis, mu_np[i] - std_np[i], mu_np[i] + std_np[i],
                                 alpha=0.15, color="blue")
            plt.xlabel("Time step"); plt.ylabel("Load")
            plt.title("All Forecasts • Mean ± 1σ (Improved ARIMA)")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(f"./result/{global_fig}", dpi=300)
            visualize = False
            # plt.show()


    num_pts = len(test_loader.dataset) * mu.size(1)
    test_mse = running_mse / len(test_loader.dataset)
    test_nll = running_nll / num_pts
    test_crps = running_crps / len(test_loader.dataset)
    print(f"\n🧪 Test MSE : {test_mse:.6f}")
    # print(f"🧪 Test NLL : {test_nll:.6f}")
    print(f"🧪 Test CRPS: {test_crps:.6f}")
    return test_mse, test_nll, test_crps


if __name__ == "__main__":
    set_seed(42)

    # ── hyper‑params ──────────────────────────────────────────────────────
    batch_size = 32
    epochs     = 1000
    lr         = 1e-2            # ↑ slightly faster convergence
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "ARIMA";
    data_name = "Residential" # Building  Spanish Consumption Residential Solar
    model_path = f"{data_name}_{model_name}_best_model.pt"
    print(f"Using device: {device}")

    # (A) Load & prepare data ------------------------------------------------
    if data_name == "Building":
        times, load, temp, workday, season = get_data_building_weather_weekly()
    elif data_name == "Spanish":
        times, load, temp, workday, season = get_data_spanish_weekly()
        epochs = 2000
    elif data_name == "Consumption":
        times, load, temp, workday, season = get_data_power_consumption_weekly()
    elif data_name == "Residential":
        times, load, temp, workday, season = get_data_residential_weekly()
    elif data_name == "Solar":
        times, load, temp, workday, season= get_data_solar_weather_weekly()

    feature_dict = dict(load=load, temp=temp, workday=workday, season=season)

    load_mean, load_std = float(load.mean()), float(load.std())

    train_data, test_data, _ = process_seq2seq_data(
        feature_dict      = feature_dict,
        train_ratio       = 0.7,
        output_len        = 3,
        encoder_len_weeks = 2,
        decoder_len_weeks = 1,
        device            = device)

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader  = make_loader(test_data, batch_size, shuffle=False)

    # ── build & train ────────────────────────────────────────────────────
    model = WeekForecastProbARIMA(p=48, forecast_len=168,  load_mean=load_mean, load_std=load_std).to(device)

    # Uncomment if you want to retrain
    import os
    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_prob_model(model, train_loader, epochs=epochs, lr=lr, device=device, save_path=model_path, weight_decay=1e-4)


    # fresh instance for clean eval
    model_eval = WeekForecastProbARIMA(p=48, forecast_len=168, load_mean=load_mean, load_std=load_std).to(device)

    evaluate_arima_model(model_eval, test_loader, device,
                             model_path=model_path,
                             visualize=True,
                             n_vis_samples=5,
                             global_fig="mean_traj_var_ARIMA.png")
