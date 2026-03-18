import torch
import torch.nn as nn
import math


# ============================================================
# Meta Components (v2)
#   - raw external tensor can still be K_ext = 29
#   - but expert count is 3:
#       1) thermal block
#       2) workday
#       3) season
# ============================================================

class MetaNet(nn.Module):
    """
    v2:
      - supports block input, not only scalar input
      - output is still [B, input_dim, xprime_dim]
      - hidden width follows the original paper-style dx * dx' = input_dim * xprime_dim
        For dx=1 and xprime_dim=40, hidden width = 40
    """
    def __init__(self, input_dim, xprime_dim, feat_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.xprime_dim = xprime_dim
        self.feat_dim = feat_dim

        hidden_dim = input_dim * xprime_dim
        self.layer1 = nn.Linear(feat_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, input_dim * xprime_dim)

    def forward(self, x_feat):
        """
        x_feat: [B, feat_dim]
        return: [B, input_dim, xprime_dim]
        """
        B = x_feat.size(0)
        out = torch.tanh(self.layer1(x_feat))
        out = torch.tanh(self.layer2(out))
        return out.view(B, self.input_dim, self.xprime_dim)


class GatingNet(nn.Module):
    """
    unchanged logic:
      gate is generated from hidden state h
      top-k sparse softmax during/after warmup
    """
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.num_experts = num_experts
        if num_experts > 0:
            self.layer1 = nn.Linear(hidden_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, num_experts)

    def forward(self, h, epoch=None, top_k=None, warmup_epochs=0):
        if getattr(self, "num_experts", 0) == 0:
            return None

        logits = self.layer2(torch.tanh(self.layer1(h)))  # [B, num_experts]

        if (epoch is None) or (top_k is None) or (epoch < warmup_epochs) or (top_k <= 0):
            return torch.softmax(logits, dim=-1)

        k = min(top_k, self.num_experts)
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        return torch.softmax(masked_logits, dim=-1)


class MetaTransformBlock(nn.Module):
    """
    v2 block-expert transform:

    Input:
      x_l   : [B, input_dim]
      x_ext : [B, K_ext]

    Two modes:
    1) block mode (recommended):
         - 3 experts: thermal / workday / season
         - thermal expert can eat multi-dim block
    2) fallback mode:
         - original scalar-per-channel expert behavior
         - used only if block indices are not provided
    """
    def __init__(
        self,
        xprime_dim,
        hidden_size,
        input_dim=1,
        n_externals=0,
        thermal_indices=None,
        workday_index=None,
        season_index=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.xprime_dim = xprime_dim
        self.n_externals = n_externals

        self.use_lora_theta = False
        r = 4
        self.lora_theta_A = nn.Parameter(torch.randn(input_dim, r) / math.sqrt(max(input_dim, 1)))
        self.lora_theta_B = nn.Parameter(torch.zeros(r, xprime_dim))

        self.theta0 = nn.Parameter(torch.zeros(1, input_dim, xprime_dim))
        self.ln = nn.LayerNorm([input_dim, xprime_dim])

        # --------------------------
        # block-expert config
        # --------------------------
        self.thermal_indices = None if thermal_indices is None else list(thermal_indices)
        self.workday_index = workday_index
        self.season_index = season_index

        self.use_block_experts = (
            self.thermal_indices is not None
            and self.workday_index is not None
            and self.season_index is not None
        )

        if self.use_block_experts:
            self.num_experts = 3

            thermal_dim = len(self.thermal_indices)
            self.meta_thermal = MetaNet(input_dim=input_dim, xprime_dim=xprime_dim, feat_dim=thermal_dim)
            self.meta_workday = MetaNet(input_dim=input_dim, xprime_dim=xprime_dim, feat_dim=1)
            self.meta_season = MetaNet(input_dim=input_dim, xprime_dim=xprime_dim, feat_dim=1)

            self.gating = GatingNet(hidden_size, num_experts=self.num_experts)

        else:
            # fallback to original behavior
            self.num_experts = n_externals
            self.meta_feats = nn.ModuleList(
                [MetaNet(input_dim=input_dim, xprime_dim=xprime_dim, feat_dim=1) for _ in range(n_externals)]
            )
            self.gating = GatingNet(hidden_size, num_experts=n_externals)

    def _forward_block_mode(self, h_prev_rnn, x_l, x_ext, epoch=None, top_k=None, warmup_epochs=0):
        B = x_l.size(0)

        thermal_feat = x_ext[:, self.thermal_indices]                              # [B, Dth]
        workday_feat = x_ext[:, self.workday_index:self.workday_index + 1]         # [B, 1]
        season_feat = x_ext[:, self.season_index:self.season_index + 1]            # [B, 1]

        W_thermal = self.ln(self.meta_thermal(thermal_feat))                       # [B, 1, xprime]
        W_workday = self.ln(self.meta_workday(workday_feat))                       # [B, 1, xprime]
        W_season = self.ln(self.meta_season(season_feat))                          # [B, 1, xprime]

        W_experts = torch.stack([W_thermal, W_workday, W_season], dim=1)           # [B, 3, 1, xprime]

        gates = self.gating(
            h_prev_rnn,
            epoch=epoch,
            top_k=top_k,
            warmup_epochs=warmup_epochs,
        )                                                                          # [B, 3]

        gates_expanded = gates.view(B, self.num_experts, 1, 1)
        theta_dynamic = (W_experts * gates_expanded).sum(dim=1)                    # [B, 1, xprime]

        if self.use_lora_theta:
            delta_theta = (self.lora_theta_A @ self.lora_theta_B).unsqueeze(0)
            theta = theta_dynamic + self.theta0 + delta_theta
        else:
            theta = theta_dynamic + self.theta0

        x_prime = torch.bmm(x_l.unsqueeze(1), theta).squeeze(1)                    # [B, xprime]
        return x_prime, theta

    def _forward_original_mode(self, h_prev_rnn, x_l, x_ext, epoch=None, top_k=None, warmup_epochs=0):
        B = x_l.size(0)

        if self.n_externals > 0:
            Ws = []
            for j in range(self.n_externals):
                xj = x_ext[:, j:j + 1]
                Wj = self.ln(self.meta_feats[j](xj))
                Ws.append(Wj)
            W_experts = torch.stack(Ws, dim=1)                                     # [B, K_ext, 1, xprime]

            gates = self.gating(
                h_prev_rnn,
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )                                                                      # [B, K_ext]

            gates_expanded = gates.view(B, self.n_externals, 1, 1)
            theta_dynamic = (W_experts * gates_expanded).sum(dim=1)
        else:
            theta_dynamic = torch.zeros(B, self.input_dim, self.xprime_dim, device=x_l.device)

        if self.use_lora_theta:
            delta_theta = (self.lora_theta_A @ self.lora_theta_B).unsqueeze(0)
            theta = theta_dynamic + self.theta0 + delta_theta
        else:
            theta = theta_dynamic + self.theta0

        x_prime = torch.bmm(x_l.unsqueeze(1), theta).squeeze(1)
        return x_prime, theta

    def forward(self, h_prev_rnn, x_l, x_ext, epoch=None, top_k=None, warmup_epochs=0):
        """
        x_l:   [B, input_dim]
        x_ext: [B, K_ext]
        """
        if self.use_block_experts:
            return self._forward_block_mode(
                h_prev_rnn, x_l, x_ext,
                epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
            )
        else:
            return self._forward_original_mode(
                h_prev_rnn, x_l, x_ext,
                epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
            )


# ============================================================
# Encoder
# ============================================================

class Encoder_meta(nn.Module):
    def __init__(self, xprime_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            xprime_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x_l_seq, x_ext_seq, transform_block,
                h_init=None, epoch=None, top_k=None, warmup_epochs=0):
        B, T, _ = x_l_seq.shape
        device = x_l_seq.device
        h_rnn = torch.zeros(self.num_layers, B, self.hidden_size, device=device) if h_init is None else h_init

        for t in range(T):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(
                h_for_meta,
                x_l_seq[:, t],
                x_ext_seq[:, t] if x_ext_seq.size(-1) > 0 else x_ext_seq[:, t:t + 1],
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )
            x_prime = x_prime.unsqueeze(1)
            _, h_rnn = self.rnn(x_prime, h_rnn)

        return h_rnn


# ============================================================
# Decoder
# ============================================================

class Decoder_meta(nn.Module):
    def __init__(self, xprime_dim, latent_size, output_len, output_dim=1,
                 num_layers=1, dropout=0.1, hidden_size=None):
        super().__init__()
        self.latent_size = latent_size
        self.output_len = output_len
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            xprime_dim,
            latent_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.head = nn.Linear(latent_size, output_len * output_dim)

        assert hidden_size is not None, "You must provide hidden_size for projection."
        self.project = nn.ModuleList([nn.Linear(hidden_size, latent_size) for _ in range(num_layers)])

    def forward(self, x_l_seq, x_ext_seq, h_init, transform_block,
                epoch=None, top_k=None, warmup_epochs=0):
        B, L, _ = x_l_seq.shape

        h_rnn = torch.stack([self.project[i](h_init[i]) for i in range(self.num_layers)], dim=0)

        preds = []

        h_last = h_rnn[-1]
        pred_0 = self.head(h_last).view(B, self.output_len, self.output_dim)
        preds.append(pred_0.unsqueeze(1))

        for t in range(L):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(
                h_for_meta,
                x_l_seq[:, t],
                x_ext_seq[:, t] if x_ext_seq.size(-1) > 0 else x_ext_seq[:, t:t + 1],
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )
            x_prime = x_prime.unsqueeze(1)
            out_t, h_rnn = self.rnn(x_prime, h_rnn)
            pred_t = self.head(out_t.squeeze(1)).view(B, self.output_len, self.output_dim)
            preds.append(pred_t.unsqueeze(1))

        preds = torch.cat(preds, dim=1)
        return preds


# ============================================================
# Full Seq2Seq (deterministic)
# ============================================================

class Seq2Seq_meta(nn.Module):
    def __init__(
        self,
        xprime_dim,
        input_dim,
        hidden_size,
        latent_size,
        output_len,
        n_externals,
        output_dim=1,
        num_layers=1,
        dropout=0.1,
        thermal_indices=None,
        workday_index=None,
        season_index=None,
    ):
        super().__init__()
        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            input_dim=input_dim,
            n_externals=n_externals,
            thermal_indices=thermal_indices,
            workday_index=workday_index,
            season_index=season_index,
        )
        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim,
            hidden_size=latent_size,
            input_dim=input_dim,
            n_externals=n_externals,
            thermal_indices=thermal_indices,
            workday_index=workday_index,
            season_index=season_index,
        )
        self.encoder = Encoder_meta(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder_meta(
            xprime_dim=xprime_dim,
            latent_size=latent_size,
            output_len=output_len,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            hidden_size=hidden_size,
        )

    def forward(self, enc_l, enc_ext, dec_l, dec_ext, epoch=None, top_k=None, warmup_epochs=0):
        h_enc = self.encoder(
            enc_l, enc_ext,
            transform_block=self.transform_enc,
            epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
        )
        preds = self.decoder(
            dec_l, dec_ext,
            h_init=h_enc,
            transform_block=self.transform_dec,
            epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
        )
        return preds


# ============================================================
# Variational Encoder / Decoder
# ============================================================

class VariationalEncoder_meta(nn.Module):
    def __init__(self, xprime_dim, hidden_size, latent_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            xprime_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)

    def forward(self, x_l_seq, x_ext_seq, transform_block,
                h_init=None, epoch=None, top_k=None, warmup_epochs=0):
        B, T, _ = x_l_seq.shape
        device = x_l_seq.device
        h_rnn = torch.zeros(self.num_layers, B, self.hidden_size, device=device) if h_init is None else h_init

        for t in range(T):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(
                h_for_meta,
                x_l_seq[:, t],
                x_ext_seq[:, t] if x_ext_seq.size(-1) > 0 else x_ext_seq[:, t:t + 1],
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )
            x_prime = x_prime.unsqueeze(1)
            _, h_rnn = self.rnn(x_prime, h_rnn)

        h_last = h_rnn[-1]
        mu = self.mu_layer(h_last)
        logvar = self.logvar_layer(h_last)
        return mu, logvar


class VariationalDecoder_meta_predvar(nn.Module):
    def __init__(self, xprime_dim, latent_size, output_len, output_dim=1,
                 num_layers=1, dropout=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.output_len = output_len
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            xprime_dim,
            latent_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.head_mu = nn.Linear(latent_size, output_len * output_dim)
        self.head_logvar = nn.Linear(latent_size, output_len * output_dim)

    def forward(self, x_l_seq, x_ext_seq, z_latent, transform_block,
                epoch=None, top_k=None, warmup_epochs=0):
        B, L, _ = x_l_seq.shape
        h_rnn = z_latent.unsqueeze(0).repeat(self.num_layers, 1, 1)

        mu_preds = []
        logvar_preds = []

        h_last = h_rnn[-1]
        mu_0 = self.head_mu(h_last).view(B, self.output_len, self.output_dim)
        logvar_0 = self.head_logvar(h_last).view(B, self.output_len, self.output_dim)
        mu_preds.append(mu_0.unsqueeze(1))
        logvar_preds.append(logvar_0.unsqueeze(1))

        for t in range(L):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(
                h_for_meta,
                x_l_seq[:, t],
                x_ext_seq[:, t] if x_ext_seq.size(-1) > 0 else x_ext_seq[:, t:t + 1],
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )
            x_prime = x_prime.unsqueeze(1)
            out_t, h_rnn = self.rnn(x_prime, h_rnn)

            mu_t = self.head_mu(out_t.squeeze(1)).view(B, self.output_len, self.output_dim)
            logvar_t = self.head_logvar(out_t.squeeze(1)).view(B, self.output_len, self.output_dim)

            mu_preds.append(mu_t.unsqueeze(1))
            logvar_preds.append(logvar_t.unsqueeze(1))

        mu_preds = torch.cat(mu_preds, dim=1)
        logvar_preds = torch.cat(logvar_preds, dim=1)
        return mu_preds, logvar_preds


# ============================================================
# Full Variational Seq2Seq
# ============================================================

class VariationalSeq2Seq_meta(nn.Module):
    def __init__(
        self,
        xprime_dim,
        input_dim,
        hidden_size,
        latent_size,
        output_len,
        n_externals,
        output_dim=1,
        num_layers=1,
        dropout=0.1,
        thermal_indices=None,
        workday_index=None,
        season_index=None,
    ):
        super().__init__()
        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            input_dim=input_dim,
            n_externals=n_externals,
            thermal_indices=thermal_indices,
            workday_index=workday_index,
            season_index=season_index,
        )
        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim,
            hidden_size=latent_size,
            input_dim=input_dim,
            n_externals=n_externals,
            thermal_indices=thermal_indices,
            workday_index=workday_index,
            season_index=season_index,
        )
        self.encoder = VariationalEncoder_meta(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = VariationalDecoder_meta_predvar(
            xprime_dim=xprime_dim,
            latent_size=latent_size,
            output_len=output_len,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, enc_l, enc_ext, dec_l, dec_ext, epoch=None, top_k=None, warmup_epochs=0):
        mu, logvar = self.encoder(
            enc_l, enc_ext,
            transform_block=self.transform_enc,
            epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
        )
        z = self.reparameterize(mu, logvar)
        mu_preds, logvar_preds = self.decoder(
            dec_l, dec_ext,
            z_latent=z,
            transform_block=self.transform_dec,
            epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
        )
        return mu_preds, logvar_preds, mu, logvar