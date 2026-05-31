import torch
import torch.nn as nn
import math
import torch.nn.functional as F  # 确保文件头部有这个 import


# ============================================================
# Meta Components (v2)
#   - raw external tensor can still be K_ext = 29
#   - but expert count is 3:
#       1) thermal block
#       2) workday
#       3) season
# ============================================================

class MetaNet(nn.Module):
    def __init__(self, input_dim, xprime_dim, feat_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.xprime_dim = xprime_dim
        self.feat_dim = feat_dim
        hidden_dim = input_dim * xprime_dim
        self.layer1 = nn.Linear(feat_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, input_dim * xprime_dim)

    def forward(self, x_feat):
        B = x_feat.size(0)
        out = F.gelu(self.layer1(x_feat)) 
        out = self.layer2(out) 
        return out.view(B, self.input_dim, self.xprime_dim)



class GatingNet(nn.Module):
    def __init__(self, gate_input_dim, num_experts, gate_hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        if num_experts > 0:
            # 如果没有指定隐藏层维度，默认和输入维度一致
            if gate_hidden_dim is None:
                gate_hidden_dim = gate_input_dim
            self.layer1 = nn.Linear(gate_input_dim, gate_hidden_dim)
            self.layer2 = nn.Linear(gate_hidden_dim, num_experts)

    def forward(self, gate_input, epoch=None, top_k=None, warmup_epochs=0):
        # gate_input 形状: [B, gate_input_dim] 或 [B*T, gate_input_dim]
        if getattr(self, "num_experts", 0) == 0:
            return None
        out = F.leaky_relu(self.layer1(gate_input), negative_slope=0.01)
        logits = self.layer2(out)
        
        if (epoch is None) or (top_k is None) or (epoch < warmup_epochs) or (top_k <= 0):
            return torch.softmax(logits, dim=-1)
        k = min(top_k, self.num_experts)
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        return torch.softmax(masked_logits, dim=-1)


class MetaTransformBlock(nn.Module):
    """
    Flexible meta-transform block.

    `expert_specs` is a list of dictionaries, for example:
        [
            {"name": "temp", "indices": [0, 6, ..., 29]},
            {"name": "workday", "indices": [1, 2]},
            {"name": "season", "indices": [3]},
        ]

    Each item becomes one meta expert. The expert input dimension is
    len(indices). This lets the main script decide how many external
    variables are used as experts without changing the model code.
    """
    def __init__(
        self,
        xprime_dim,
        hidden_size,
        input_dim=1,
        n_externals=0,
        expert_specs=None,
        # Backward-compatible arguments. They are ignored if expert_specs is provided.
        thermal_indices=None,
        workday_index=None,
        season_index=None,
        holiday_index=None,
        month_indices=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.xprime_dim = xprime_dim
        self.n_externals = n_externals

        self.use_lora_theta = False
        r = 4
        self.lora_theta_A = nn.Parameter(torch.randn(input_dim, r) / math.sqrt(max(input_dim, 1)))
        self.lora_theta_B = nn.Parameter(torch.zeros(r, xprime_dim))
        self.theta0 = nn.Parameter(torch.empty(1, input_dim, xprime_dim))
        nn.init.xavier_normal_(self.theta0)

        # Preferred flexible interface.
        if expert_specs is not None:
            self.expert_specs = [
                {"name": str(e["name"]), "indices": list(e["indices"])}
                for e in expert_specs
            ]
        else:
            # Backward-compatible fallback for older calls.
            self.expert_specs = []
            if thermal_indices is not None:
                self.expert_specs.append({"name": "temp", "indices": list(thermal_indices)})
            if workday_index is not None:
                self.expert_specs.append({"name": "workday", "indices": [workday_index]})
            if season_index is not None:
                self.expert_specs.append({"name": "season", "indices": [season_index]})
            if holiday_index is not None:
                self.expert_specs.append({"name": "holiday", "indices": [holiday_index]})
            if month_indices is not None:
                self.expert_specs.append({"name": "month", "indices": list(month_indices)})

        self.use_block_experts = len(self.expert_specs) > 0
        gate_input_dim = input_dim + n_externals

        if self.use_block_experts:
            self.num_experts = len(self.expert_specs)
            self.meta_experts = nn.ModuleList([
                MetaNet(
                    input_dim=input_dim,
                    xprime_dim=xprime_dim,
                    feat_dim=len(spec["indices"]),
                )
                for spec in self.expert_specs
            ])
            self.ln_experts = nn.ModuleList([
                nn.LayerNorm([input_dim, xprime_dim])
                for _ in self.expert_specs
            ])
            self.gating = GatingNet(gate_input_dim=gate_input_dim, num_experts=self.num_experts)
        else:
            self.num_experts = n_externals
            self.meta_feats = nn.ModuleList(
                [MetaNet(input_dim=input_dim, xprime_dim=xprime_dim, feat_dim=1) for _ in range(n_externals)]
            )
            self.ln = nn.LayerNorm([input_dim, xprime_dim])
            self.gating = GatingNet(gate_input_dim=gate_input_dim, num_experts=n_externals)

    def _forward_block_mode(self, h_prev_rnn, x_l, x_ext, epoch=None, top_k=None, warmup_epochs=0):
        B = x_l.size(0)

        W_list = []
        for spec, meta, ln in zip(self.expert_specs, self.meta_experts, self.ln_experts):
            feat = x_ext[:, spec["indices"]]
            W_list.append(ln(meta(feat)))

        W_experts = torch.stack(W_list, dim=1)  # [B, num_experts, input_dim, xprime_dim]

        # Use the same gate input as the vectorized path: current load + all external features.
        gate_input = torch.cat([x_l, x_ext], dim=-1)
        gates = self.gating(
            gate_input,
            epoch=epoch,
            top_k=top_k,
            warmup_epochs=warmup_epochs,
        )  # [B, num_experts]

        gates_expanded = gates.view(B, self.num_experts, 1, 1)
        theta_dynamic = (W_experts * gates_expanded).sum(dim=1)

        if self.use_lora_theta:
            delta_theta = (self.lora_theta_A @ self.lora_theta_B).unsqueeze(0)
            theta = theta_dynamic + self.theta0 + delta_theta
        else:
            theta = theta_dynamic + self.theta0

        x_prime = torch.bmm(x_l.unsqueeze(1), theta).squeeze(1)
        return x_prime, theta

    def _forward_original_mode(self, h_prev_rnn, x_l, x_ext, epoch=None, top_k=None, warmup_epochs=0):
        B = x_l.size(0)

        if self.n_externals > 0:
            Ws = []
            for j in range(self.n_externals):
                xj = x_ext[:, j:j + 1]
                Wj = self.ln(self.meta_feats[j](xj))
                Ws.append(Wj)
            W_experts = torch.stack(Ws, dim=1)

            gate_input = torch.cat([x_l, x_ext], dim=-1)
            gates = self.gating(
                gate_input,
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )
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

    def forward_batch_seq(self, x_l_seq, x_ext_seq, epoch=None, top_k=None, warmup_epochs=0):
        """
        x_l_seq:   [B, T, input_dim]
        x_ext_seq: [B, T, n_externals]
        return: x_prime_seq [B, T, xprime_dim]
        """
        B, T, _ = x_l_seq.shape

        gate_input = torch.cat([x_l_seq, x_ext_seq], dim=-1)
        gate_input_flat = gate_input.reshape(B * T, -1)

        x_ext_flat = x_ext_seq.reshape(B * T, -1)
        x_l_flat = x_l_seq.reshape(B * T, self.input_dim)

        if self.use_block_experts:
            W_list = []
            for spec, meta, ln in zip(self.expert_specs, self.meta_experts, self.ln_experts):
                feat = x_ext_flat[:, spec["indices"]]
                W_list.append(ln(meta(feat)))

            W_experts = torch.stack(W_list, dim=1)  # [B*T, num_experts, input_dim, xprime_dim]

            gates = self.gating(
                gate_input_flat,
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )  # [B*T, num_experts]
            gates_expanded = gates.view(B * T, self.num_experts, 1, 1)

            theta_dynamic = (W_experts * gates_expanded).sum(dim=1)
            theta = theta_dynamic + self.theta0
            if self.use_lora_theta:
                delta_theta = (self.lora_theta_A @ self.lora_theta_B).unsqueeze(0)
                theta = theta + delta_theta

            x_prime_flat = torch.bmm(x_l_flat.unsqueeze(1), theta).squeeze(1)
            return x_prime_flat.view(B, T, self.xprime_dim)
        else:
            if self.n_externals > 0:
                Ws = []
                for j in range(self.n_externals):
                    xj = x_ext_flat[:, j:j + 1]
                    Wj = self.ln(self.meta_feats[j](xj))
                    Ws.append(Wj)
                W_experts = torch.stack(Ws, dim=1)
                gates = self.gating(gate_input_flat, epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs)
                gates_expanded = gates.view(B * T, self.n_externals, 1, 1)
                theta_dynamic = (W_experts * gates_expanded).sum(dim=1)
            else:
                theta_dynamic = torch.zeros(B * T, self.input_dim, self.xprime_dim, device=x_l_seq.device)

            theta = theta_dynamic + self.theta0
            if self.use_lora_theta:
                delta_theta = (self.lora_theta_A @ self.lora_theta_B).unsqueeze(0)
                theta = theta + delta_theta
            x_prime_flat = torch.bmm(x_l_flat.unsqueeze(1), theta).squeeze(1)
            return x_prime_flat.view(B, T, self.xprime_dim)


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
                x_ext_seq[:, t],
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
                x_ext_seq[:, t],
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
        holiday_index=None,
        month_indices=None,
        expert_specs=None,
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
            holiday_index=holiday_index,
            month_indices=month_indices,
            expert_specs=expert_specs,
        )
        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim,
            hidden_size=latent_size,
            input_dim=input_dim,
            n_externals=n_externals,
            thermal_indices=thermal_indices,
            workday_index=workday_index,
            season_index=season_index,
            holiday_index=holiday_index,
            month_indices=month_indices,
            expert_specs=expert_specs,
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

    def forward(self, enc_l, enc_ext, dec_l, dec_ext, epoch=None, top_k=None, warmup_epochs=0, forecast_indices=None):
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
        self.rnn = nn.GRU(xprime_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)

    def forward(self, x_l_seq, x_ext_seq, transform_block, h_init=None, epoch=None, top_k=None, warmup_epochs=0):
        B, T, _ = x_l_seq.shape
        device = x_l_seq.device
        h_rnn = torch.zeros(self.num_layers, B, self.hidden_size, device=device) if h_init is None else h_init

        # 【加速核心】一次性算出所有 x_prime，然后一把塞给 GRU
        x_prime_seq = transform_block.forward_batch_seq(
            x_l_seq, x_ext_seq, epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
        )
        _, h_rnn = self.rnn(x_prime_seq, h_rnn)

        h_last = h_rnn[-1]
        mu = self.mu_layer(h_last)
        logvar = self.logvar_layer(h_last)
        return mu, logvar


class VariationalDecoder_meta_predvar(nn.Module):
    def __init__(self, xprime_dim, latent_size, output_len, output_dim=1, num_layers=1, dropout=0.1, logvar_min=-9.0, logvar_max=-3.0):
        super().__init__()
        self.latent_size = latent_size
        self.output_len = output_len  # 24
        self.output_dim = output_dim  # 1
        self.num_layers = num_layers
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # ---------------------------------------------------------
        # 1. 原有的滚动步 GRU (处理 145 步的 x_prime_seq)
        # ---------------------------------------------------------
        self.rnn = nn.GRU(xprime_dim, latent_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # ---------------------------------------------------------
        # 2. 【新增核心】专属的 Hourly GRU (用于自回归解码 24h)
        # 输入: 上一个小时的预测均值 mu (维度=output_dim)
        # 隐层: 继承自滚动步 GRU 的隐状态 (维度=latent_size)
        # ---------------------------------------------------------
        self.hourly_gru = nn.GRU(
            input_size=output_dim, 
            hidden_size=latent_size, 
            num_layers=1, # 通常1层足够，避免过拟合
            batch_first=True
        )

        # ---------------------------------------------------------
        # 3. 【修改】单步输出头 (不再是 output_len * output_dim，而是 output_dim)
        # ---------------------------------------------------------
        self.head_mu = nn.Linear(latent_size, output_dim)
        self.head_logvar = nn.Linear(latent_size, output_dim)

        self.project = nn.ModuleList([nn.Linear(latent_size, latent_size) for _ in range(num_layers)])

    def _decode_24h(self, h_init):
        """
        使用 Hourly GRU 自回归地解码 24 小时的 mu 和 logvar
        h_init: [B, latent_size] 或 [B*L, latent_size]
        返回: mu_24h [B, 24, 1], logvar_24h [B, 24, 1]
        """
        B = h_init.size(0)
        device = h_init.device
        
        # 初始化 Hourly GRU 的隐状态: [1, B, latent_size]
        h_hourly = h_init.unsqueeze(0)
        
        # 初始输入: 全零向量作为 <GO> token, [B, 1, output_dim]
        input_k = torch.zeros(B, 1, self.output_dim, device=device)
        
        mu_list = []
        logvar_list = []
        
        for k in range(self.output_len):
            # 逐步递推
            out_k, h_hourly = self.hourly_gru(input_k, h_hourly) 
            # out_k: [B, 1, latent_size]
            
            # 预测当前小时的 mu 和 logvar
            mu_k = self.head_mu(out_k.squeeze(1))           # [B, output_dim]
            raw_logvar_k = self.head_logvar(out_k.squeeze(1))
            logvar_k = self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(raw_logvar_k)
            
            mu_list.append(mu_k)
            logvar_list.append(logvar_k)
            
            # 下一步的输入是当前步预测的 mu (Autoregressive)
            input_k = mu_k.unsqueeze(1) 
            
        # 拼接 24 个小时
        mu_24h = torch.stack(mu_list, dim=1)       # [B, 24, output_dim]
        logvar_24h = torch.stack(logvar_list, dim=1) # [B, 24, output_dim]
        
        return mu_24h, logvar_24h

    def forward(self, x_l_seq, x_ext_seq, z_latent, transform_block,
                epoch=None, top_k=None, warmup_epochs=0, forecast_indices=None):
        B, L, _ = x_l_seq.shape

        # 1. Initialize hidden state from latent variable.
        h_rnn_init = torch.stack([self.project[i](z_latent) for i in range(self.num_layers)], dim=0)
        h_last_init = h_rnn_init[-1]  # [B, latent_size], forecast origin t=0

        # 2. Still roll the hidden state through every decoder hour, because
        # origin t=24 must have absorbed dec_l[0:24].
        x_prime_seq = transform_block.forward_batch_seq(
            x_l_seq, x_ext_seq, epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs
        )
        out_seq, _ = self.rnn(x_prime_seq, h_rnn_init)  # [B, L, latent_size]

        # 3. Only decode selected forecast origins. For day-ahead protocol,
        # forecast_indices = [0, 24, 48, 72, 96, 120, 144].
        if forecast_indices is None:
            selected = list(range(L + 1))
        elif torch.is_tensor(forecast_indices):
            selected = [int(i) for i in forecast_indices.detach().cpu().tolist()]
        else:
            selected = [int(i) for i in forecast_indices]

        if len(selected) == 0:
            raise ValueError("forecast_indices is empty; no forecasts would be produced.")
        bad = [i for i in selected if i < 0 or i > L]
        if bad:
            raise ValueError(f"forecast_indices must be in [0, {L}], got invalid entries {bad}.")

        hidden_list = []
        for origin in selected:
            if origin == 0:
                hidden_list.append(h_last_init)
            else:
                # after consuming x_l_seq[:, :origin], i.e., out_seq at origin-1
                hidden_list.append(out_seq[:, origin - 1, :])

        hidden_stack = torch.stack(hidden_list, dim=1)  # [B, M, latent_size]
        M = hidden_stack.size(1)
        hidden_flat = hidden_stack.reshape(B * M, -1)

        mu_flat, logvar_flat = self._decode_24h(hidden_flat)
        mu_preds = mu_flat.view(B, M, self.output_len, self.output_dim)
        logvar_preds = logvar_flat.view(B, M, self.output_len, self.output_dim)

        return mu_preds, logvar_preds

# ============================================================
# Full Variational Seq2Seq
# ============================================================

class VariationalSeq2Seq_meta(nn.Module):
    def __init__(
        self, xprime_dim, input_dim, hidden_size, latent_size, output_len, 
        n_externals, output_dim=1, num_layers=1, dropout=0.1, 
        thermal_indices=None, workday_index=None, season_index=None, holiday_index=None,
        month_indices=None, expert_specs=None,
        logvar_min=-9.0, logvar_max=-3.0  # 【新增】接收超参数
    ):
        super().__init__()
        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim, hidden_size=hidden_size, input_dim=input_dim, 
            n_externals=n_externals, thermal_indices=thermal_indices, 
            workday_index=workday_index, season_index=season_index, holiday_index=holiday_index,
            month_indices=month_indices,
            expert_specs=expert_specs,
        )
        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim, hidden_size=latent_size, input_dim=input_dim, 
            n_externals=n_externals, thermal_indices=thermal_indices, 
            workday_index=workday_index, season_index=season_index, holiday_index=holiday_index,
            month_indices=month_indices,
            expert_specs=expert_specs,
        )
        self.encoder = VariationalEncoder_meta(
            xprime_dim=xprime_dim, hidden_size=hidden_size, latent_size=latent_size, 
            num_layers=num_layers, dropout=dropout,
        )
        self.decoder = VariationalDecoder_meta_predvar(
            xprime_dim=xprime_dim, latent_size=latent_size, output_len=output_len, 
            output_dim=output_dim, num_layers=num_layers, dropout=dropout,
            logvar_min=logvar_min, logvar_max=logvar_max  # 【新增】传给 Decoder
        )

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, enc_l, enc_ext, dec_l, dec_ext, epoch=None, top_k=None, warmup_epochs=0, forecast_indices=None):
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
            epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs,
            forecast_indices=forecast_indices
        )
        return mu_preds, logvar_preds, mu, logvar