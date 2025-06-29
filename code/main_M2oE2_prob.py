import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from data_utils import *
from model import *
import numpy as np, random, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import math

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(model, train_loader, epochs, lr, device, save_path="best_model.pt"):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_train_loss = float("inf")
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            (enc_l, enc_t, enc_w, enc_s,
             dec_l, dec_t, dec_w, dec_s,
             tgt) = [t.to(device) for t in batch]

            optimizer.zero_grad()

            mu_preds, logvar_preds, mu_z, logvar_z = model(enc_l, enc_t, enc_w, enc_s,
                                                           dec_l, dec_t, dec_w, dec_s,
                                                           epoch=ep,
                                                           top_k=top_k, warmup_epochs=10)

            nll = gaussian_nll_loss(mu_preds, logvar_preds, tgt)
            kl = kl_loss(mu_z, logvar_z)

            loss = nll + 0.01 * kl

            # reconstruction_loss = nn.functional.mse_loss(preds, tgt, reduction='mean')
            # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # loss = reconstruction_loss + kl_weight * kl_loss  # KL weight is tunable

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * enc_l.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = ep
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {ep} with loss {best_train_loss:.6f}")

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | Train MSE: {avg_train_loss:.6f} | Best MSE: {best_train_loss:.6f} (epoch {best_epoch})")

    print(f"\n🏁 Training completed. Best model saved from epoch {best_epoch} with MSE: {best_train_loss:.6f}")
    return model



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


@torch.no_grad()
def evaluate_model(model, test_loader, loss_fn, device,
                   model_path="model.pt", reduce="first", visualize=True):
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    running_mse = 0.0
    running_nll = 0.0
    running_crps = 0.0

    for batch in test_loader:
        (enc_l, enc_t, enc_w, enc_s,
         dec_l, dec_t, dec_w, dec_s,
         tgt) = [t.to(device) for t in batch]

        B = enc_l.size(0)

        mu_preds, logvar_preds, _, _ = model(enc_l, enc_t, enc_w, enc_s,
                                             dec_l, dec_t, dec_w, dec_s)
        mu_preds = mu_preds.squeeze(-1)          # [B, L+1, output_len]
        logvar_preds = logvar_preds.squeeze(-1)  # [B, L+1, output_len]
        tgt = tgt.squeeze(-1)                    # [B, L+1, output_len]

        if reduce == "mean":
            for b in range(B):
                pred_avg = reconstruct_sequence(mu_preds[b])  # [L+output_len]
                tgt_avg = reconstruct_sequence(tgt[b])
                all_preds.append(pred_avg.cpu())
                all_targets.append(tgt_avg.cpu())
                running_mse += loss_fn(pred_avg, tgt_avg).item()

        elif reduce == "first":
            mu_first = mu_preds[:, :, 0]           # [B, L+1]
            logvar_first = logvar_preds[:, :, 0]   # [B, L+1]
            tgt_first = tgt[:, :, 0]               # [B, L+1]

            all_preds.extend(mu_first.cpu())
            all_targets.extend(tgt_first.cpu())
            running_mse += loss_fn(mu_first, tgt_first).item() * B

            # NLL
            nll = 0.5 * (
                logvar_first +
                torch.log(torch.tensor(2 * np.pi, device=logvar_first.device)) +
                (tgt_first - mu_first) ** 2 / logvar_first.exp()
            )  # [B, L+1]
            running_nll += nll.sum().item()

            # CRPS
            crps = crps_gaussian(mu_first, logvar_first, tgt_first)
            running_crps += crps.item() * B

            # Visualization
            if visualize:
                for i in range(min(5, mu_first.size(0))):
                    std_pred = logvar_first[i].exp().sqrt().cpu()
                    plt.figure(figsize=(4, 2))
                    plt.plot(tgt_first[i].cpu(), label='True', linestyle='--', color='red')
                    plt.plot(mu_first[i].cpu(), label='Mean Predicted', alpha=0.6,  color='blue',)
                    plt.fill_between(np.arange(mu_first.size(1)),
                                     mu_first[i].cpu() - std_pred,
                                     mu_first[i].cpu() + std_pred,
                                     color='blue', alpha=0.1, label='±1 Std Predicted')
                    # plt.title(f"Prediction + Uncertainty (Sample {i})")
                    # plt.legend()
                    plt.ylim(0, 1)
                    plt.yticks([0, 0.5, 1], fontsize=14)
                    plt.xticks(fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"./result/{data_name}_{model_name}_sample_{i}.pdf")

                    # handles, labels = plt.gca().get_legend_handles_labels()
                    # plt.legend(handles, labels,
                    #            ncol=len(labels),  # one long row
                    #            loc='upper center',  # put it where you like
                    #            bbox_to_anchor=(0.5, 1.05),# and nudge it above the axes
                    #            framealpha=1,
                    #            fontsize= 14
                    #            )
                    plt.show()

                # Global visualization
                plt.figure(figsize=(12, 6))
                for i in range(mu_first.size(0)):
                    std_pred = logvar_first[i].exp().sqrt().cpu()
                    plt.plot(tgt_first[i].cpu(), color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
                    plt.plot(mu_first[i].cpu(), linewidth=2.0, label='Mean Pred' if i == 0 else None)
                    plt.fill_between(np.arange(mu_first.size(1)),
                                     mu_first[i].cpu() - std_pred,
                                     mu_first[i].cpu() + std_pred,
                                     alpha=0.2, color='red')
                plt.title("All Forecasts: Mean + Predicted Variance")
                plt.xlabel("Time step")
                plt.ylabel("Forecasted value")
                plt.legend(loc='upper right')
                plt.tight_layout()
                visualize = False
                # plt.show()
        else:
            raise ValueError("reduce must be 'mean' or 'first'")

    test_mse = running_mse / len(test_loader.dataset)
    test_nll = running_nll / (len(test_loader.dataset) * mu_first.size(1)) if reduce == "first" else None
    test_crps = running_crps / len(test_loader.dataset) if reduce == "first" else None

    print(f"🧪 Test MSE: {test_mse:.6f}")
    # print(f"🧪 Test NLL : {test_nll:.6f}")
    print(f"🧪 Test CRPS: {test_crps:.6f}")

    return test_mse, test_nll, test_crps


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    seed        = 42
    set_seed(seed)
    batch_size  = 16
    epochs      = 300
    lr          = 1e-3
    kl_weight   = 0.01
    xprime_dim  = 40
    hidden_dim  = 64
    latent_dim  = 32
    num_layers  = 4
    output_len  = 3            # make sure this matches process_seq2seq_data
    num_experts = 3 # temp, workday, season
    top_k = 2
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_name = "Solar"  # Spanish Consumption Residential Solar
    model_name = "M2OE2"
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

    input_dim = 1
    output_dim = 1  # predict one-dimensional load


    feature_dict = dict(load    = load,
                        temp    = temp,
                        workday = workday,
                        season  = season)

    train_data, test_data, _ = process_seq2seq_data(
        feature_dict     = feature_dict,
        train_ratio      = 0.7,
        output_len       = output_len,
        device           = device)

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader  = make_loader(test_data,  batch_size, shuffle=False)

    model = VariationalSeq2Seq_meta(
        xprime_dim=xprime_dim,
        input_dim=input_dim,
        hidden_size=hidden_dim,
        latent_size=latent_dim,
        output_len=output_len,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1,
        num_experts=num_experts
    ).to(device)

    import os
    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device, save_path=model_path)

    # Re-initialize the model with same architecture
    model = VariationalSeq2Seq_meta(
        xprime_dim=xprime_dim,
        input_dim=input_dim,
        hidden_size=hidden_dim,
        latent_size=latent_dim,
        output_len=output_len,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1,
        num_experts=num_experts
    ).to(device)

    # Then evaluate
    evaluate_model(model, test_loader, nn.MSELoss(), device, model_path=model_path)
