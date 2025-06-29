import torch
import torch.nn as nn
import numpy as np

# ---------------- Meta Components ----------------
class MetaNet(nn.Module):
    def __init__(self, input_dim, xprime_dim):
        super().__init__()
        self.layer1 = nn.Linear(1, input_dim * xprime_dim)
        self.layer2 = nn.Linear(input_dim * xprime_dim, input_dim * xprime_dim)
        self.input_dim = input_dim
        self.xprime_dim = xprime_dim

    def forward(self, x_feat):  # x_feat: [B, 1]
        B = x_feat.size(0)
        out = torch.tanh(self.layer1(x_feat))            # [B, 32]
        out = torch.tanh(self.layer2(out))               # [B, input_dim * xprime_dim]
        return out.view(B, self.input_dim, self.xprime_dim)  # [B, input_dim, xprime_dim]



class GatingNet(nn.Module):
    def __init__(self, hidden_size, num_experts=3):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_experts)

    def forward(self, h, epoch=None, top_k=None, warmup_epochs=0):
        logits = self.layer2(torch.tanh(self.layer1(h)))  # [B, num_experts]

        if (epoch is None) or (top_k is None) or (epoch < warmup_epochs):
            return torch.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(masked_logits, dim=-1)


class MetaTransformBlock(nn.Module):
    def __init__(self, xprime_dim, num_experts=3, input_dim=1, hidden_size=64):
        super().__init__()
        self.meta_temp = MetaNet(input_dim, xprime_dim)
        self.meta_work = MetaNet(input_dim, xprime_dim)
        self.meta_season = MetaNet(input_dim, xprime_dim)
        self.gating = GatingNet(hidden_size, num_experts)  # Use hidden_size here
        self.ln = nn.LayerNorm([input_dim, xprime_dim])
        self.theta0 = nn.Parameter(torch.zeros(1, input_dim, xprime_dim))

    def forward(self, h_prev_rnn, x_l, x_t, x_w, x_s, epoch=None, top_k=None, warmup_epochs=0):
        w_temp = self.ln(self.meta_temp(x_t))     # [B, input_dim, xprime_dim]
        w_work = self.ln(self.meta_work(x_w))     # [B, input_dim, xprime_dim]
        w_seas = self.ln(self.meta_season(x_s))   # [B, input_dim, xprime_dim]

        gates = self.gating(h_prev_rnn, epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)  # [B, num_experts]
        W_experts = torch.stack([w_temp, w_work, w_seas], dim=1)  # [B, num_experts, input_dim, xprime_dim]
        gates_expanded = gates.view(gates.size(0), gates.size(1), 1, 1)  # [B, num_experts, 1, 1]
        theta_dynamic = (W_experts * gates_expanded).sum(dim=1)  # [B, input_dim, xprime_dim]
        theta = theta_dynamic + self.theta0                      # [B, input_dim, xprime_dim]

        x_prime = torch.bmm(x_l.unsqueeze(1), theta).squeeze(1)  # [B, xprime_dim]
        return x_prime, theta

# ---------------- Encoder ----------------
class Encoder_meta(nn.Module):
    def __init__(self, xprime_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(xprime_dim, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

    def forward(self, x_l_seq, x_t_seq, x_w_seq, x_s_seq,
                transform_block, h_init=None, epoch=None, top_k=None, warmup_epochs=0):
        B, T, _ = x_l_seq.shape
        h_rnn = torch.zeros(self.num_layers, B, self.hidden_size, device=x_l_seq.device) if h_init is None else h_init

        for t in range(T):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(h_for_meta,
                                         x_l_seq[:, t], x_t_seq[:, t],
                                         x_w_seq[:, t], x_s_seq[:, t],
                                         epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
            x_prime = x_prime.unsqueeze(1)
            _, h_rnn = self.rnn(x_prime, h_rnn)

        return h_rnn  # [num_layers, B, hidden_size]


# ---------------- Decoder ----------------
class Decoder_meta(nn.Module):
    def __init__(self, xprime_dim, latent_size, output_len, output_dim=1,
                 num_layers=1, dropout=0.1, hidden_size=None):
        super().__init__()
        self.latent_size = latent_size
        self.output_len = output_len
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(xprime_dim, latent_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.head = nn.Linear(latent_size, output_len * output_dim)

        # Layer-wise projection from encoder hidden_size → decoder latent_size
        assert hidden_size is not None, "You must provide hidden_size for projection."
        self.project = nn.ModuleList([
            nn.Linear(hidden_size, latent_size) for _ in range(num_layers)
        ])

    def forward(self, x_l_seq, x_t_seq, x_w_seq, x_s_seq,
                h_init, transform_block,
                epoch=None, top_k=None, warmup_epochs=0):
        B, L, _ = x_l_seq.shape

        # Project each layer of encoder hidden state to latent size
        h_rnn = torch.stack([
            self.project[i](h_init[i]) for i in range(self.num_layers)
        ], dim=0)  # [num_layers, B, latent_size]

        preds = []

        # Step 0
        h_last = h_rnn[-1]  # [B, latent_size]
        pred_0 = self.head(h_last).view(B, self.output_len, self.output_dim)
        preds.append(pred_0.unsqueeze(1))  # [B, 1, output_len, output_dim]

        # Steps 1 to L
        for t in range(L):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(h_for_meta,
                                         x_l_seq[:, t], x_t_seq[:, t],
                                         x_w_seq[:, t], x_s_seq[:, t],
                                         epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
            x_prime = x_prime.unsqueeze(1)
            out_t, h_rnn = self.rnn(x_prime, h_rnn)
            pred_t = self.head(out_t.squeeze(1)).view(B, self.output_len, self.output_dim)
            preds.append(pred_t.unsqueeze(1))

        preds = torch.cat(preds, dim=1)  # [B, L+1, output_len, output_dim]
        return preds


# ---------------- Full Seq2Seq Model ----------------
class Seq2Seq_meta(nn.Module):
    def __init__(self, xprime_dim, input_dim, hidden_size, latent_size,
                 output_len, output_dim=1, num_layers=1, dropout=0.1, num_experts=3):
        super().__init__()

        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim,
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_size=hidden_size  # encoder hidden_size
        )

        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim,
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_size=latent_size  # decoder latent_size
        )

        self.encoder = Encoder_meta(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)

        self.decoder = Decoder_meta(
            xprime_dim=xprime_dim,
            latent_size=latent_size,
            output_len=output_len,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            hidden_size=hidden_size  # for projection from encoder hidden
        )

    def forward(self,
                enc_l, enc_t, enc_w, enc_s,
                dec_l, dec_t, dec_w, dec_s,
                epoch=None, top_k=None, warmup_epochs=0):

        h_enc = self.encoder(enc_l, enc_t, enc_w, enc_s,
                             transform_block=self.transform_enc,
                             epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)

        preds = self.decoder(dec_l, dec_t, dec_w, dec_s,
                             h_init=h_enc,
                             transform_block=self.transform_dec,
                             epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
        return preds



# ---------------- Encoder ----------------
class VariationalEncoder_meta(nn.Module):
    def __init__(self, xprime_dim, hidden_size, latent_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(xprime_dim, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)

    def forward(self, x_l_seq, x_t_seq, x_w_seq, x_s_seq,
                transform_block, h_init=None, epoch=None, top_k=None, warmup_epochs=0):

        B, T, _ = x_l_seq.shape
        h_rnn = torch.zeros(self.num_layers, B, self.hidden_size, device=x_l_seq.device) if h_init is None else h_init

        for t in range(T):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(h_for_meta,
                                         x_l_seq[:, t], x_t_seq[:, t],
                                         x_w_seq[:, t], x_s_seq[:, t],
                                         epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
            x_prime = x_prime.unsqueeze(1)
            _, h_rnn = self.rnn(x_prime, h_rnn)

        h_last = h_rnn[-1]  # [B, hidden_size]
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

        self.rnn = nn.GRU(xprime_dim, latent_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        # Separate heads for mean and log-variance
        self.head_mu = nn.Linear(latent_size, output_len * output_dim)
        self.head_logvar = nn.Linear(latent_size, output_len * output_dim)

    def forward(self, x_l_seq, x_t_seq, x_w_seq, x_s_seq,
                z_latent, transform_block,
                epoch=None, top_k=None, warmup_epochs=0):
        B, L, _ = x_l_seq.shape

        h_rnn = z_latent.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, latent_size]

        mu_preds = []
        logvar_preds = []

        # Step 0
        h_last = h_rnn[-1]
        mu_0 = self.head_mu(h_last).view(B, self.output_len, self.output_dim)
        logvar_0 = self.head_logvar(h_last).view(B, self.output_len, self.output_dim)
        mu_preds.append(mu_0.unsqueeze(1))           # [B, 1, output_len, output_dim]
        logvar_preds.append(logvar_0.unsqueeze(1))   # same shape

        # Steps 1 to L
        for t in range(L):
            h_for_meta = h_rnn[-1]
            x_prime, _ = transform_block(h_for_meta,
                                         x_l_seq[:, t], x_t_seq[:, t],
                                         x_w_seq[:, t], x_s_seq[:, t],
                                         epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
            x_prime = x_prime.unsqueeze(1)
            out_t, h_rnn = self.rnn(x_prime, h_rnn)

            mu_t = self.head_mu(out_t.squeeze(1)).view(B, self.output_len, self.output_dim)
            logvar_t = self.head_logvar(out_t.squeeze(1)).view(B, self.output_len, self.output_dim)

            mu_preds.append(mu_t.unsqueeze(1))
            logvar_preds.append(logvar_t.unsqueeze(1))

        # Stack across time
        mu_preds = torch.cat(mu_preds, dim=1)         # [B, L+1, output_len, output_dim]
        logvar_preds = torch.cat(logvar_preds, dim=1) # same shape

        return mu_preds, logvar_preds


# ---------------- Full Seq2Seq Model ----------------
class VariationalSeq2Seq_meta(nn.Module):
    def __init__(self, xprime_dim, input_dim, hidden_size, latent_size,
                 output_len, output_dim=1, num_layers=1, dropout=0.1, num_experts=3):
        super().__init__()

        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim,
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_size=hidden_size  # encoder hidden size
        )

        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim,
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_size=latent_size  # decoder latent size
        )

        self.encoder = VariationalEncoder_meta(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # self.decoder = VariationalDecoder_meta_fixvar(
        #     xprime_dim=xprime_dim,
        #     latent_size=latent_size,
        #     output_len=output_len,
        #     output_dim=output_dim,
        #     num_layers=num_layers,
        #     dropout=dropout
        # )

        self.decoder = VariationalDecoder_meta_predvar(
            xprime_dim=xprime_dim,
            latent_size=latent_size,
            output_len=output_len,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                enc_l, enc_t, enc_w, enc_s,
                dec_l, dec_t, dec_w, dec_s,
                epoch=None, top_k=None, warmup_epochs=0):

        mu, logvar = self.encoder(enc_l, enc_t, enc_w, enc_s,
                                  transform_block=self.transform_enc,
                                  epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)

        z = self.reparameterize(mu, logvar)  # [B, latent_size]

        mu_preds, logvar_preds = self.decoder(dec_l, dec_t, dec_w, dec_s,
                             z_latent=z,
                             transform_block=self.transform_dec,
                             epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)

        return mu_preds, logvar_preds, mu, logvar






# # ---------------- Decoder v1: fixed variance ----------------
# class VariationalDecoder_meta_fixvar(nn.Module):
#     def __init__(self, xprime_dim, latent_size, output_len, output_dim=1,
#                  num_layers=1, dropout=0.1, fixed_var_value=0.01):
#         super().__init__()
#         self.latent_size = latent_size
#         self.output_len = output_len
#         self.output_dim = output_dim
#         self.num_layers = num_layers
#
#         self.rnn = nn.GRU(xprime_dim, latent_size, num_layers,
#                           batch_first=True,
#                           dropout=dropout if num_layers > 1 else 0)
#
#         self.head = nn.Linear(latent_size, output_len * output_dim)
#
#         # Fixed log-variance (scalar)
#         self.fixed_logvar = torch.tensor(np.log(fixed_var_value), dtype=torch.float32)
#
#     def forward(self, x_l_seq, x_t_seq, x_w_seq, x_s_seq,
#                 z_latent, transform_block,
#                 epoch=None, top_k=None, warmup_epochs=0):
#         B, L, _ = x_l_seq.shape
#
#         h_rnn = z_latent.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, latent_size]
#
#         mu_preds = []
#
#         # Step 0
#         h_last = h_rnn[-1]
#         mu_0 = self.head(h_last).view(B, self.output_len, self.output_dim)
#         mu_preds.append(mu_0.unsqueeze(1))  # [B, 1, output_len, output_dim]
#
#         # Steps 1 to L
#         for t in range(L):
#             h_for_meta = h_rnn[-1]
#             x_prime, _ = transform_block(h_for_meta,
#                                          x_l_seq[:, t], x_t_seq[:, t],
#                                          x_w_seq[:, t], x_s_seq[:, t],
#                                          epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
#             x_prime = x_prime.unsqueeze(1)
#             out_t, h_rnn = self.rnn(x_prime, h_rnn)
#
#             mu_t = self.head(out_t.squeeze(1)).view(B, self.output_len, self.output_dim)
#             mu_preds.append(mu_t.unsqueeze(1))
#
#         mu_preds = torch.cat(mu_preds, dim=1)  # [B, L+1, output_len, output_dim]
#
#         # Now create logvar_preds: same shape, filled with fixed_logvar
#         logvar_preds = self.fixed_logvar.expand_as(mu_preds).to(mu_preds.device)
#
#         return mu_preds, logvar_preds
#


# ---------------- Decoder v2: predicted variance ----------------


#
# ## LSTM
# import torch, torch.nn as nn
# import torch.nn.functional as F
#
# class LSTM_Baseline(nn.Module):
#     """
#     Simple encoder‑decoder LSTM baseline.
#     • All four modal inputs (load, temp, workday, season) are concatenated along feature dim
#       so the external information is still available, but the model is otherwise “plain”.
#     • The forward signature (extra **kwargs) lets the old training loop pass epoch/top_k/warmup
#       without breaking anything.
#     """
#     def __init__(
#         self,
#         input_dim: int,      # 1  →  only the scalar value of each channel
#         hidden_size: int,    # e.g. 64
#         output_len: int,     # prediction horizon (3)
#         output_dim: int = 1, # scalar prediction
#         num_layers: int = 2,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.hidden_size  = hidden_size
#         self.output_len   = output_len
#         self.output_dim   = output_dim
#         self.num_layers   = num_layers
#
#         # encoder & decoder
#         self.encoder = nn.LSTM(
#             input_size = input_dim * 4,    # four channels concatenated
#             hidden_size = hidden_size,
#             num_layers = num_layers,
#             batch_first = True,
#             dropout = dropout if num_layers > 1 else 0.0,
#         )
#         self.decoder = nn.LSTM(
#             input_size = input_dim * 4,
#             hidden_size = hidden_size,
#             num_layers = num_layers,
#             batch_first = True,
#             dropout = dropout if num_layers > 1 else 0.0,
#         )
#
#         self.out_layer = nn.Linear(hidden_size, output_dim)
#
#     def forward(
#         self,
#         enc_l, enc_t, enc_w, enc_s,
#         dec_l, dec_t, dec_w, dec_s,
#         *unused, **unused_kw,
#     ):
#         """
#         enc_* : [B, Lenc, 1]      (load / temp / workday / season)
#         dec_* : [B, Ldec, 1]
#         return: [B, Lenc+1, output_len, 1]  (to keep your downstream code intact)
#         """
#         B, Lenc, _ = enc_l.shape
#
#         # 1) ---------- Encode ----------
#         enc_in = torch.cat([enc_l, enc_t, enc_w, enc_s], dim=-1)   # [B, Lenc, 4]
#         _, (h_n, c_n) = self.encoder(enc_in)                       # carry hidden to decoder
#
#         # 2) ---------- Decode ----------
#         Ldec = dec_l.size(1)                                        # usually 1 step (the teacher‑force token)
#         dec_in = torch.cat([dec_l, dec_t, dec_w, dec_s], dim=-1)    # [B, Ldec, 4]
#         dec_out, _ = self.decoder(dec_in, (h_n, c_n))               # [B, Ldec, H]
#         y0 = self.out_layer(dec_out[:, -1])                         # last step → [B, output_dim]
#
#         # 3) ---------- Autoregressive forecast ----------
#         preds = []
#         ht, ct = h_n, c_n
#         xt = dec_in[:, -1]                                          # start token
#         for _ in range(self.output_len):
#             xt = xt.unsqueeze(1)                                    # [B,1,4]
#             out, (ht, ct) = self.decoder(xt, (ht, ct))              # [B,1,H]
#             yt = self.out_layer(out.squeeze(1))                     # [B, output_dim]
#             preds.append(yt)
#             # next decoder input = last prediction repeated over 4 channels
#             xt = torch.cat([yt]*4, dim=-1)
#
#         # 3) ---------- Autoregressive forecast ----------
#         preds = torch.stack(preds, dim=1)  # [B, H, 1]
#
#         # 4) ---------- match original return shape ----------
#         seq_len_y = enc_l.size(1) - self.output_len + 1  # <-- NEW: 168‑>166
#         preds = preds.unsqueeze(1).repeat(1, seq_len_y, 1, 1)
#         return preds  # [B, 166, 3, 1]
#
