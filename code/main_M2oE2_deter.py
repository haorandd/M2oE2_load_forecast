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

            output = model(enc_l, enc_t, enc_w, enc_s,
                           dec_l, dec_t, dec_w, dec_s,
                           epoch=ep,
                           top_k=top_k,  # Only 1 expert used
                           warmup_epochs=10)  # No sparsity for first 10 epochs

            loss = loss_fn(output, tgt)

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

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def evaluate_model(model, test_loader, loss_fn, device,
                   model_path="model.pt", reduce="mean", visualize=True):
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    running_loss = 0.0

    for batch in test_loader:
        (enc_l, enc_t, enc_w, enc_s,
         dec_l, dec_t, dec_w, dec_s,
         tgt) = [t.to(device) for t in batch]

        pred = model(enc_l, enc_t, enc_w, enc_s,
                     dec_l, dec_t, dec_w, dec_s)  # [B, L+1, output_len, 1]
        pred = pred.squeeze(-1)  # [B, L+1, output_len]
        tgt = tgt.squeeze(-1)    # [B, L+1, output_len]

        B = pred.size(0)

        if reduce == "mean":
            for b in range(B):
                pred_avg = reconstruct_sequence(pred[b])  # [L+output_len]
                tgt_avg = reconstruct_sequence(tgt[b])    # same shape

                all_preds.append(pred_avg.cpu())
                all_targets.append(tgt_avg.cpu())
                running_loss += loss_fn(pred_avg, tgt_avg).item()
        elif reduce == "first":
            pred_first = pred[:, :, 0]  # [B, L+1]
            tgt_first = tgt[:, :, 0]    # [B, L+1]

            all_preds.extend(pred_first.cpu())
            all_targets.extend(tgt_first.cpu())
            running_loss += loss_fn(pred_first, tgt_first).item() * B
        else:
            raise ValueError("reduce must be 'mean' or 'first'")

    test_mse = running_loss / len(test_loader.dataset)
    print(f"\n🧪 Test MSE: {test_mse:.6f}")

    if visualize:
        all_preds = torch.stack(all_preds)
        all_targets = torch.stack(all_targets)

        for i in range(min(5, all_preds.size(0))):
            plt.figure(figsize=(8, 2))
            plt.plot(all_targets[i], label='True', linestyle='--')
            plt.plot(all_preds[i], label='Pred', alpha=0.8)
            plt.title(f"Prediction vs. True (Sample {i})")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return test_mse




# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    seed        = 42
    batch_size  = 16
    epochs      = 200
    lr          = 1e-3
    xprime_dim  = 40
    hidden_dim  = 64
    latent_dim  = 32
    num_layers  = 3
    output_len  = 3            # make sure this matches process_seq2seq_data
    num_experts = 3 # temp, workday, season
    top_k = 2
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Building_M2OE2_best_model.pt"

    set_seed(seed)
    print(f"Using device: {device}")

    # (A) Load & prepare data ------------------------------------------------
    times, load, temp, workday, season = get_data_building_weather_weekly()

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

    model = Seq2Seq_meta(
        xprime_dim=xprime_dim,  # dimension of x_prime after meta transform
        input_dim=input_dim,  # dimension of original input x_l
        hidden_size=hidden_dim,  # encoder RNN hidden size
        latent_size=latent_dim,  # decoder RNN hidden size
        output_len=output_len,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1,
        num_experts=num_experts
    ).to(device)

    train_model(model, train_loader, epochs=epochs, lr=lr, device=device, save_path=model_name)

    # Re-initialize the model with same architecture
    model = Seq2Seq_meta(
        xprime_dim=xprime_dim,  # dimension of x_prime after meta transform
        input_dim=input_dim,  # dimension of original input x_l
        hidden_size=hidden_dim,  # encoder RNN hidden size
        latent_size=latent_dim,  # decoder RNN hidden size
        output_len=output_len,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1,
        num_experts=num_experts
    ).to(device)

    # Then evaluate
    evaluate_model(model, test_loader, nn.MSELoss(), device, model_path=model_name)
