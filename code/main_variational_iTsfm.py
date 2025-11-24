import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_iTransformer import Model as ITransformerModel  # adjust path if needed
from model_v1 import MetaTransformBlock
from data_utils import *
import numpy as np, random, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import math

# --------------------------------------------
# Probabilistic Meta-representation + i-Transformer
# --------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep your MetaTransformBlock as-is
# from <your_module> import MetaTransformBlock

from utils_iTransformer import Model as ITransformerModel  # leave it untouched


class ITransCfg:
    def __init__(self,
                 seq_len, pred_len,
                 d_model=128, n_heads=8, e_layers=2, d_ff=256,
                 factor=5, dropout=0.1, activation="gelu",
                 use_norm=True, output_attention=False,
                 class_strategy="projection", embed="fixed", freq="h"):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.use_norm = use_norm
        self.output_attention = output_attention
        self.class_strategy = class_strategy
        self.embed = embed
        self.freq = freq


class CausalConv1d(nn.Module):
    """Causal 1D conv for per-timestep meta-gating context (no recurrence)."""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=0)
        self.k = k
    def forward(self, x):  # x: [B, T, C]
        x = x.transpose(1, 2)               # [B, C, T]
        x = F.pad(x, (self.k - 1, 0))       # left pad = causal
        x = self.conv(x)                    # [B, out_ch, T]
        return x.transpose(1, 2)            # [B, T, out_ch]


class MetaITransformerProb(nn.Module):
    """
    Outputs Gaussian parameters for each step:
      mu_preds:     [B, L+1, H, D]
      logvar_preds: [B, L+1, H, D]
    Also returns (mu_z, logvar_z) as zeros so your KL term is 0 (no code changes).
    """
    def __init__(self,
                 xprime_dim,      # tokens per timestep (becomes N in i-Transformer)
                 input_dim,       # 1 for 'load'
                 hidden_size,     # gating context (encoder side)
                 latent_size,     # gating context (decoder side)
                 output_len,      # horizon H
                 n_externals,
                 output_dim=1,
                 itr_cfg: ITransCfg = None,
                 kl_dim: int = 32,          # dims for dummy KL (zeros -> KL~0)
                 logvar_clip=(-10.0, 5.0)): # numeric stability for log-variance
        super().__init__()
        assert itr_cfg is not None, "Provide ITransCfg(seq_len, pred_len) for i-Transformer."
        assert itr_cfg.pred_len == output_len, "itr_cfg.pred_len must equal output_len."

        # Meta blocks (unchanged)
        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim, hidden_size=hidden_size,
            input_dim=input_dim, n_externals=n_externals
        )
        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim, hidden_size=latent_size,
            input_dim=input_dim, n_externals=n_externals
        )

        # Causal contexts for gating
        self.ctx_enc = CausalConv1d(in_ch=input_dim, out_ch=hidden_size, k=3)
        self.ctx_dec = CausalConv1d(in_ch=input_dim, out_ch=latent_size,  k=3)

        # i-Transformer (unchanged)
        self.itr = ITransformerModel(itr_cfg)
        self.seq_len  = itr_cfg.seq_len
        self.pred_len = itr_cfg.pred_len

        # Probabilistic heads: map tokens (N=xprime_dim) -> output_dim
        self.head_mu     = nn.Linear(xprime_dim, output_dim)
        self.head_logvar = nn.Linear(xprime_dim, output_dim)
        self.logvar_min, self.logvar_max = logvar_clip

        # Dummy latent stats for KL (zeros => KL≈0 in your kl_loss)
        self.kl_dim = kl_dim

    @torch.no_grad()
    def _ensure_len(self, x, want_len):
        """Left-pad or clip along time (dim=1) to fixed want_len."""
        B, T, C = x.shape
        if T == want_len:
            return x
        if T < want_len:
            pad = x.new_zeros(B, want_len - T, C)
            return torch.cat([pad, x], dim=1)
        return x[:, -want_len:, :]

    @staticmethod
    def _pre_norm_outofplace(x):
        # x: [B, T, N]
        means = x.mean(dim=1, keepdim=True)                     # [B,1,N]
        x_center = x - means                                    # out-of-place
        stdev = torch.sqrt(x_center.var(dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B,1,N]
        x_norm = x_center / stdev                               # out-of-place
        return x_norm, means, stdev

    @staticmethod
    def _post_denorm_outofplace(y, means, stdev):
        # y: [B, H, N]; means/stdev: [B,1,N]
        return y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    def _meta_map_over_window(self, x_l_win, x_ext_win, use_decoder_ctx=False,
                              epoch=None, top_k=None, warmup_epochs=0):
        """
        Build x' for a given window.
        x_l_win:   [B, Tw, 1]
        x_ext_win: [B, Tw, K]
        """
        B, Tw, _ = x_l_win.shape
        h_seq = (self.ctx_dec(x_l_win) if use_decoder_ctx else self.ctx_enc(x_l_win))  # [B, Tw, S]
        xprime_seq = []
        block = self.transform_dec if use_decoder_ctx else self.transform_enc
        for t in range(Tw):
            x_t = x_l_win[:, t]                         # [B,1]
            e_t = x_ext_win[:, t] if x_ext_win.size(-1) > 0 else x_ext_win[:, t:t+1]
            h_t = h_seq[:, t]                           # [B,S]
            x_prime, _ = block(h_t, x_t, e_t,
                               epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
            xprime_seq.append(x_prime.unsqueeze(1))
        return torch.cat(xprime_seq, dim=1)             # [B, Tw, xprime_dim]

    def forward(self, enc_l, enc_ext, dec_l, dec_ext,
                epoch=None, top_k=None, warmup_epochs=0):
        """
        enc_l:   [B, Te, 1]
        enc_ext: [B, Te, K]
        dec_l:   [B, L,  1]
        dec_ext: [B, L,  K]
        Returns:
          mu_preds, logvar_preds, mu_z, logvar_z
            mu_preds/logvar_preds: [B, L+1, H, D]
            mu_z/logvar_z:         [B, kl_dim] zeros (so KL≈0 with your kl_loss)
        """
        B, Te, _ = enc_l.shape
        L = dec_l.size(1)

        # Precompute encoder meta-map once (saves compute)
        xprime_enc = self._meta_map_over_window(
            enc_l, enc_ext, use_decoder_ctx=False,
            epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
        )  # [B, Te, xprime_dim]

        mu_steps, logv_steps = [], []

        for s in range(L + 1):
            if s == 0:
                xprime_hist = xprime_enc
            else:
                xprime_dec = self._meta_map_over_window(
                    dec_l[:, :s], dec_ext[:, :s], use_decoder_ctx=True,
                    epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
                )                                       # [B, s, xprime_dim]
                xprime_hist = torch.cat([xprime_enc, xprime_dec], dim=1)  # [B, Te+s, N]

            xprime_fixed = self._ensure_len(xprime_hist, self.seq_len)    # [B, seq_len, N]

            # --- NEW: safe normalization outside the i-Transformer ---
            x_norm, means, stdev = self._pre_norm_outofplace(xprime_fixed) # all out-of-place

            # i-Transformer forward on normalized input (use_norm=False in cfg)
            itr_out = self.itr(x_norm, None, None, None)                    # [B, H, N]

            # denormalize outputs to original scale (out-of-place)
            itr_out = self._post_denorm_outofplace(itr_out, means, stdev)   # [B, H, N]
            # ---------------------------------------------------------

            # i-Transformer forward: returns [B, pred_len, N]
            # itr_out = self.itr(xprime_fixed, None, None, None)            # no time marks used

            # Heads: [B, pred_len, N] -> [B, pred_len, D]
            mu_t     = self.head_mu(itr_out)                               # [B, H, D]
            logvar_t = self.head_logvar(itr_out)                           # [B, H, D]
            logvar_t = torch.clamp(logvar_t, min=self.logvar_min, max=self.logvar_max)

            mu_steps.append(mu_t.unsqueeze(1))         # [B,1,H,D]
            logv_steps.append(logvar_t.unsqueeze(1))   # [B,1,H,D]

        mu_preds     = torch.cat(mu_steps,   dim=1)     # [B, L+1, H, D]
        logvar_preds = torch.cat(logv_steps, dim=1)     # [B, L+1, H, D]

        # Dummy latent stats so your train loop (KL) works unchanged
        mu_z     = enc_l.new_zeros(B, self.kl_dim)
        logvar_z = enc_l.new_zeros(B, self.kl_dim)

        return mu_preds, logvar_preds, mu_z, logvar_z




def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- helpers ----------
def gaussian_icdf(p, device):
    # Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        2 * torch.as_tensor(p, device=device) - 1
    )

def pinball_loss(y, yq, q):
    # elementwise pinball (quantile) loss; lower is better
    e = y - yq
    return torch.where(e >= 0, q * e, (q - 1) * e)

def winkler_score(y, L, U, alpha):
    # elementwise Winkler score for two-sided (1-alpha) interval; lower is better
    width = (U - L)
    below = (L - y).clamp(min=0.0)
    above = (y - U).clamp(min=0.0)
    return width + (2.0 / alpha) * (below + above)


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


def make_loader(split_dict, batch_size, shuffle):
    ds = TensorDataset(
        split_dict['X_enc_l'],
        split_dict['X_enc_ext'],
        split_dict['X_dec_in_l'],
        split_dict['X_dec_in_ext'],
        split_dict['Y_dec_target'],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def apply_random_missing(d, rate=0.3, missing_value=float('nan')):
    """
    In-place: add random missingness to input tensors in dict `d`.
    Creates companion masks (1 = observed, 0 = missing).
    Does NOT touch Y_dec_target.
    """
    keys_to_mask = ['X_enc_l', 'X_dec_in_l', 'X_enc_ext', 'X_dec_in_ext']
    device = None
    for k in keys_to_mask:
        if k not in d:
            continue
        x = d[k]
        device = x.device
        if x.numel() == 0:
            # e.g., no externals -> shape [..., 0]
            d[f'{k}_mask'] = torch.ones_like(x, dtype=torch.float32, device=device)
            continue

        # element-wise Bernoulli: 1=observed, 0=missing
        keep = (torch.rand_like(x, dtype=torch.float32) > rate).float()
        # stash mask (same shape as x)
        d[f'{k}_mask'] = keep

        # apply missingness
        if np.isnan(missing_value):
            # Set missing entries to NaN
            x_masked = x.clone()
            x_masked[keep == 0] = float('nan')
        else:
            # Set missing entries to a constant (e.g., 0.0)
            x_masked = x * keep + (1.0 - keep) * missing_value

        d[k] = x_masked

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

    unseen_case =  False
    missing_case = False
    ## unseen data case:
    if unseen_case:
        train_dict = {k: v[: int(0.3* B)] for k, v in data_tensors.items()}
        test_dict  = {k: v[int(0.7* B):] for k, v in data_tensors.items()}
    ## randomly missing case (apply to TRAIN ONLY, exact 30% per sample)
    if missing_case:
        miss_rate = 0.30
        enc_keep = max(1, int(np.ceil((1.0 - miss_rate) * enc_seq_len)))  # e.g. ceil(0.7*168)=118
        dec_keep = max(1, int(np.ceil((1.0 - miss_rate) * L)))  # e.g. ceil(0.7*165)=116
        # helper: index-select along time dim=1 with the same kept indices for all samples
        def _subselect_time(x, idx):
            # x: [B, T, ...], idx: [T_kept] on same device
            return torch.index_select(x, dim=1, index=idx)
        # choose kept indices ONCE (same positions for all training samples to keep a fixed shape)
        # If you prefer per-sample different positions, we’d need padding—keeping it fixed as requested.
        device_ = train_dict['X_enc_l'].device
        enc_idx = torch.randperm(enc_seq_len, device=device_)[:enc_keep].sort().values
        dec_idx = torch.randperm(L, device=device_)[:dec_keep].sort().values
        # targets have indices 0..L inclusive; keep the same dec_idx plus the last index L
        y_idx = torch.cat([dec_idx, torch.tensor([L], device=device_)], dim=0).unique(sorted=True)
        # apply to encoder inputs
        train_dict['X_enc_l'] = _subselect_time(train_dict['X_enc_l'], enc_idx)
        train_dict['X_enc_ext'] = _subselect_time(train_dict['X_enc_ext'], enc_idx)
        # apply to decoder inputs
        train_dict['X_dec_in_l'] = _subselect_time(train_dict['X_dec_in_l'], dec_idx)
        train_dict['X_dec_in_ext'] = _subselect_time(train_dict['X_dec_in_ext'], dec_idx)
        # apply to targets (time axis is the 2nd dim)
        train_dict['Y_dec_target'] = _subselect_time(train_dict['Y_dec_target'], y_idx)
        # ---------------------------------------------------------------------------
        # Shapes after sub-selection (for sanity)
        print("After missing (TRAIN):")
        for k, v in train_dict.items():
            print(f"{k:15s} {tuple(v.shape)}")

    return train_dict, test_dict, scalers


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(model, train_loader, epochs, lr, device, top_k=2, kl_weight=0.01, warmup_epochs=10, save_path="best_model.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_train = float("inf")
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for (enc_l, enc_ext, dec_l, dec_ext, tgt) in train_loader:
            enc_l, enc_ext, dec_l, dec_ext, tgt = enc_l.to(device), enc_ext.to(device), dec_l.to(device), dec_ext.to(device), tgt.to(device)
            optimizer.zero_grad()

            mu_preds, logvar_preds, mu_z, logvar_z = model(
                enc_l, enc_ext, dec_l, dec_ext,
                epoch=ep, top_k=top_k, warmup_epochs=warmup_epochs
            )
            nll = gaussian_nll_loss(mu_preds, logvar_preds, tgt)
            kl  = kl_loss(mu_z, logvar_z)
            loss = nll + kl_weight * kl
            loss.backward()
            optimizer.step()

            running += loss.item() * enc_l.size(0)

        avg = running / len(train_loader.dataset)
        if avg < best_train:
            best_train = avg
            best_epoch = ep
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {ep} | loss {best_train:.6f}")

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | train loss: {avg:.6f} | best: {best_train:.6f} (ep {best_epoch})")

    print(f"\n🏁 Done. Best epoch {best_epoch} | loss {best_train:.6f}")
    return model


@torch.no_grad()
def evaluate_model(model, test_loader, loss_fn, device,
                   model_path="model.pt", reduce="first", visualize=True,
                   quantiles=(0.1, 0.5, 0.9),   # for pinball loss
                   alpha=0.1                    # Winkler for (1-alpha) PI; 0.1 -> 90%
                   ):
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    running_mse = 0.0
    running_nll = 0.0
    running_crps = 0.0
    running_qpin = 0.0
    running_wink = 0.0

    # --- PVE additions (accumulators) ---
    running_pve_abs = 0.0    # sum over samples of |pred_peak - true_peak|
    running_pve_pct = 0.0    # sum over samples of |pred_peak - true_peak| / true_peak

    for batch in test_loader:
        if len(batch) == 5:
            enc_l, enc_ext, dec_l, dec_ext, tgt = [t.to(device) for t in batch]
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} tensors.")
        # ====================================

        B = enc_l.size(0)

        # ==== CHANGED: call model with new inputs ====
        mu_preds, logvar_preds, _, _ = model(enc_l, enc_ext, dec_l, dec_ext)
        print(mu_preds.shape)
        # ============================================

        mu_preds = mu_preds.squeeze(-1)          # [B, L+1, output_len]
        logvar_preds = logvar_preds.squeeze(-1)  # [B, L+1, output_len]
        tgt = tgt.squeeze(-1)                    # [B, L+1, output_len]

        if reduce == "mean":
            # NOTE: in this branch you reconstruct a single series (no per-step logvar kept),
            # so NLL/CRPS/Quantile/Winkler are not computed here.
            for b in range(B):
                pred_avg = reconstruct_sequence(mu_preds[b])  # [L+output_len]
                tgt_avg = reconstruct_sequence(tgt[b])
                all_preds.append(pred_avg.cpu())
                all_targets.append(tgt_avg.cpu())
                running_mse += loss_fn(pred_avg, tgt_avg).item()

                # --- PVE additions (per-sample) ---
                p_true = float(torch.max(tgt_avg))
                p_pred = float(torch.max(pred_avg))
                diff = abs(p_pred - p_true)
                running_pve_abs += diff
                running_pve_pct += diff / (p_true + 1e-12)

        elif reduce == "first":
            mu_first = mu_preds[:, :, 0]           # [B, L+1]
            logvar_first = logvar_preds[:, :, 0]   # [B, L+1]
            tgt_first = tgt[:, :, 0]               # [B, L+1]
            sigma_first = logvar_first.exp().sqrt()

            all_preds.extend(mu_first.cpu())
            all_targets.extend(tgt_first.cpu())
            running_mse += loss_fn(mu_first, tgt_first).item() * B

            # --- PVE additions (vectorized over batch) ---
            p_true, _ = tgt_first.max(dim=1)    # [B]
            p_pred, _ = mu_first.max(dim=1)     # [B]
            diff = (p_pred - p_true).abs()      # [B]
            running_pve_abs += diff.sum().item()
            running_pve_pct += (diff / (p_true + 1e-12)).sum().item()

            # NLL
            nll = 0.5 * (
                logvar_first +
                torch.log(torch.tensor(2 * np.pi, device=logvar_first.device)) +
                (tgt_first - mu_first) ** 2 / logvar_first.exp()
            )  # [B, L+1]
            running_nll += nll.sum().item()

            # CRPS
            crps = crps_gaussian(mu_first, logvar_first, tgt_first)  # mean over batch
            running_crps += crps.item() * B

            # Quantile (pinball) loss
            q_losses = []
            for q in quantiles:
                zq = gaussian_icdf(q, device=mu_first.device)    # scalar
                yq = mu_first + sigma_first * zq                 # predicted q-quantile
                ql = pinball_loss(tgt_first, yq, q).mean()       # avg over B*(L+1)
                q_losses.append(ql)
            qpin_mean = torch.stack(q_losses).mean()
            running_qpin += qpin_mean.item() * B

            # Winkler score
            z = gaussian_icdf(1.0 - alpha/2.0, device=mu_first.device)
            Lb = mu_first - z * sigma_first
            Ub = mu_first + z * sigma_first
            ws = winkler_score(tgt_first, Lb, Ub, alpha).mean()    # avg over B*(L+1)
            running_wink += ws.item() * B

            # Visualization (unchanged)
            if visualize:
                for i in range(min(5, mu_first.size(0))):
                    std_pred = sigma_first[i].cpu()
                    plt.figure(figsize=(4, 2))
                    plt.plot(tgt_first[i].cpu(), label='True', linestyle='--', color='red')
                    plt.plot(mu_first[i].cpu(), label='Mean Predicted', alpha=0.6, color='blue')
                    plt.fill_between(np.arange(mu_first.size(1)),
                                     mu_first[i].cpu() - std_pred,
                                     mu_first[i].cpu() + std_pred,
                                     color='blue', alpha=0.1, label='±1 Std Predicted')
                    plt.ylim(0, 1)
                    plt.yticks([0, 0.5, 1], fontsize=14)
                    plt.xticks(fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"./result/{data_name}_{model_name}_sample_{i}.pdf")

                # (2) individual plots with historical data
                for i in range(min(5, mu_first.size(0))):
                    std_pred = sigma_first[i].cpu()
                    mu_i = mu_first[i].cpu()
                    y_true_i = tgt_first[i].cpu()
                    hist_i = enc_l[i].cpu().squeeze(-1)  # history from encoder input

                    Lh = len(hist_i)
                    H  = len(mu_i)
                    x_hist = np.arange(Lh)
                    x_fore = np.arange(Lh, Lh + H)

                    plt.figure(figsize=(10, 2.5))
                    plt.plot(x_hist, hist_i, color='black', linewidth=1.5, label='History')
                    plt.plot(x_fore, y_true_i, '--', color='red', linewidth=1.5, label='True')
                    plt.plot(x_fore, mu_i, color='blue', alpha=0.8, linewidth=1.5, label='Mean Pred')
                    plt.fill_between(
                        x_fore, mu_i - std_pred, mu_i + std_pred,
                        color='blue', alpha=0.1, label='±1 σ (pred.)'
                    )
                    plt.axvline(Lh - 1, color='grey', linestyle='--', alpha=0.6)
                    plt.xlim(0, Lh + H)
                    plt.ylim(0, 1)
                    plt.tight_layout()
                    plt.yticks([0, 0.5, 1], fontsize=14)
                    plt.xticks(fontsize=14)
                    plt.legend()
                    plt.savefig(f"./result/{data_name}_{model_name}_sample_history_{i}.pdf")
                    plt.show()

                # Global visualization
                plt.figure(figsize=(12, 6))
                for i in range(mu_first.size(0)):
                    std_pred = sigma_first[i].cpu()
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
        else:
            raise ValueError("reduce must be 'mean' or 'first'")

    # --- finalize metrics ---
    test_mse = running_mse / len(test_loader.dataset)

    if reduce == "first":
        horizon = mu_first.size(1)
        test_nll  = running_nll  / (len(test_loader.dataset) * horizon)
        test_crps = running_crps / len(test_loader.dataset)
        test_qpin = running_qpin / len(test_loader.dataset)
        test_wink = running_wink / len(test_loader.dataset)
    else:
        test_nll = test_crps = test_qpin = test_wink = None

    # --- PVE finalize/print (keeps return signature unchanged) ---
    test_pve_abs = running_pve_abs / len(test_loader.dataset)
    test_pve_pct = running_pve_pct / len(test_loader.dataset)

    print(f"🧪 Test MSE         : {test_mse:.6f}")
    print(f"🧪 Test CRPS        : {test_crps:.6f}")
    if reduce == "first":
        print(f"🧪 Test QuantileLoss: {test_qpin:.6f}  (avg over q={list(quantiles)})")
        print(f"🧪 Test WinklerScore: {test_wink:.6f}  (alpha={alpha}, {(1-alpha)*100:.0f}% PI)")
    # New PVE lines
    print(f"🧪 Peak Value Error : {test_pve_abs:.6f} (absolute)")
    print(f"🧪 Peak Value Error%: {100.0*test_pve_pct:.2f}% of true peak")

    return test_mse, test_nll, test_crps, test_qpin, test_wink






# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Hyperparameters ----
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
    output_len  = 3
    top_k       = 2
    warmup_ep   = 10
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Build your feature dict (flexible) ----
    data_name = "Solar"  # Spanish Building Consumption Residential Solar Oncor_load ETTH1  ETTH2  GEFCom2014, flores
    model_name = f"M2OE2_iTsfm_v1_{output_len}hours"
    model_path = f"{data_name}_{model_name}_best_model.pt"
    print(f"Using device: {device}")

    # (A) Load & prepare data ------------------------------------------------
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

    input_dim  = 1
    output_dim = 1
    encoder_len_weeks = 1
    train_data, test_data, _ = process_seq2seq_data(
        feature_dict     = feature_dict,
        train_ratio      = 0.7,
        output_len       = output_len,
        encoder_len_weeks = encoder_len_weeks,  #  1, 4
        device           = device)

    n_externals = train_data['X_enc_ext'].shape[-1]
    print(f"K_ext (number of external features) = {n_externals}")

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    test_loader  = make_loader(test_data,  batch_size, shuffle=False)

    num_in_week = 168
    itr_cfg = ITransCfg(
        seq_len=encoder_len_weeks * num_in_week,  # e.g., 168
        pred_len=output_len,
        d_model=128, n_heads=8, e_layers=2, d_ff=256,
        dropout=0.1, activation="gelu",
        use_norm=False,  # <-- turn off internal in-place norm
        output_attention=False, class_strategy="projection",
        embed="fixed", freq="h"
    )

    model = MetaITransformerProb(
        xprime_dim=xprime_dim,
        input_dim=1,
        hidden_size=hidden_dim,  # encoder meta-gating size
        latent_size=latent_dim,  # decoder meta-gating size
        output_len=output_len,
        n_externals=train_data['X_enc_ext'].shape[-1],
        output_dim=1,
        itr_cfg=itr_cfg,
        kl_dim=latent_dim,  # so it's consistent with your prior code
        logvar_clip=(-10.0, 5.0)
    ).to(device)

    if not os.path.isfile(model_path):
        print(f"[x] Not Found '{model_path}', training.")
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device,
                    top_k=top_k, kl_weight=kl_weight, warmup_epochs=warmup_ep, save_path=model_path)
    else:
        print(f"[✓] Found '{model_path}', loading weights.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        # todo: train as well
        # train_model(model, train_loader, epochs=epochs, lr=lr, device=device,
        #             top_k=top_k, kl_weight=kl_weight, warmup_epochs=warmup_ep, save_path=model_path)
        model.eval()

    # Then evaluate
    import time
    time1 = time.time()
    evaluate_model(model, test_loader, nn.MSELoss(), device, model_path=model_path)
    time2 = time.time()
    print("time", time2-time1)