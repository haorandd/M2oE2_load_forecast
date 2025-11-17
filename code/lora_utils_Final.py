# lora_utils.py
import math
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

# Full Rank Adaptation
class DeltaLinear(nn.Module):
    """
    Implements a full-rank update (Delta) for a Linear layer.
    
    This class freezes the original weight matrix W0 and trains a delta matrix (Delta W) 
    of the same shape.
    
    Forward pass: y = x @ (W0 + scale * Delta W)^T + bias
    """
    def __init__(self, linear: nn.Linear, alpha: float = 1.0, train_bias: bool = False):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.scale = alpha

        # Freeze the original weights
        self.weight0 = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        # Trainable full-matrix increment (Delta)
        self.delta   = nn.Parameter(torch.zeros_like(self.weight0))
        
        # Handle bias training configuration
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=train_bias)
        else:
            self.bias = None

    def forward(self, x):
        W = self.weight0 + self.scale * self.delta
        y = x @ W.t()
        if self.bias is not None:
            y = y + self.bias
        return y

    def adapter_parameters(self):
        ps = [self.delta]
        if self.bias is not None and self.bias.requires_grad:
            ps.append(self.bias)
        return ps

# Low Rank Adaptation (LoRA)
class LoRALinear(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) for a Linear layer.
    
    This class freezes the original weight W0 and only trains low-rank matrices A and B.
    
    Forward pass: y = x @ W0^T + scale * x @ A^T @ B^T (+ bias)
    """
    def __init__(self, linear: nn.Linear, r=8, alpha=None, lora_dropout=0.0, train_bias=False):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.scale = (alpha if alpha is not None else r) / float(r)
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.train_bias = train_bias 

        # Freeze pre-trained weight W0; handle bias trainability
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=train_bias)

        # Initialize LoRA parameters: A ~ N(0, sigma^2), B = 0
        self.A = nn.Parameter(torch.randn(r, self.in_features) / math.sqrt(self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

    def forward(self, x):
        y = x @ self.weight.t()
        y = y + self.scale * (self.dropout(x) @ self.A.t() @ self.B.t())
        if self.bias is not None:
            y = y + self.bias
        return y

    def lora_parameters(self):
        ps = [self.A, self.B]
        if self.bias is not None and self.bias.requires_grad:
            ps.append(self.bias)
        return ps

class DeltaParam(nn.Module):
    """
    Applies a full-matrix increment to any arbitrary parameter tensor W.
    
    Equation: W_eff = W0 + scale * Delta
    """
    def __init__(self, W: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.scale = alpha
        self.delta = nn.Parameter(torch.zeros_like(W))
        W.requires_grad_(False)  # Freeze original parameter

    def forward(self, W):
        return W + self.scale * self.delta

def add_delta_to_parameter(module: nn.Module, param_name: str, alpha: float = 1.0):
    """
    Registers a Delta parametrization to a specific parameter within a module.
    """
    W = getattr(module, param_name)
    parametrize.register_parametrization(module, param_name, DeltaParam(W, alpha=alpha))
    return module

def freeze_all(module: nn.Module):
    """
    Freezes all parameters in the given module by setting requires_grad to False.
    """
    for p in module.parameters():
        p.requires_grad = False

def collect_lora_params(module: nn.Module, include_bias: bool = False):
    """
    Collects all LoRA-related parameters from the module.
    
    - Always collects A and B matrices from LoRALinear or LoRAParam.
    - If include_bias=True and the module allows bias training, collects the bias.
    """
    params = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            params += [m.A, m.B]
            if include_bias and (getattr(m, "train_bias", False)) and (m.bias is not None):
                params.append(m.bias)
        elif isinstance(m, LoRAParam):
            params += [m.A, m.B]
    return params

class LoRAParam(nn.Module):
    """
    Parametrization class that applies LoRA updates to an arbitrary weight tensor.
    """
    def __init__(self, W: torch.Tensor, r=8, alpha=None):
        super().__init__()
        out_dim, in_dim = W.shape
        self.r = r
        self.scale = (alpha if alpha is not None else r) / float(r)
        
        # Initialize A ~ N(0, sigma^2), B = 0
        A = torch.randn(r, in_dim, dtype=W.dtype, device=W.device) / math.sqrt(in_dim)
        B = torch.zeros(out_dim, r, dtype=W.dtype, device=W.device)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        
        # Freeze original weight W0
        W.requires_grad_(False)

    def forward(self, W):
        return W + self.scale * (self.B @ self.A)

def add_lora_to_parameter(module: nn.Module, param_name: str, r=8, alpha=None):
    """
    Registers a LoRA parametrization to a specific parameter within a module.
    """
    W = getattr(module, param_name) # e.g., module.theta0
    parametrize.register_parametrization(module, param_name, LoRAParam(W, r=r, alpha=alpha))
    return module

def collect_adapter_params(module: nn.Module):
    """
    Collects all adapter parameters including:
    - LoRA parameters (A, B)
    - DeltaLinear parameters (Delta W)
    - DeltaParam parameters (Delta)
    """
    params = []
    
    # Collect from explicit module classes
    for m in module.modules():
        if isinstance(m, LoRALinear):
            params += list(m.lora_parameters())
        if isinstance(m, DeltaLinear):
            params += list(m.adapter_parameters())
        
        # Check for parameters registered via parametrization (e.g., DeltaParam)
        # These are not always exposed directly in modules()
        for _, param in getattr(m, '_parameters', {}).items():
            if param is None:
                continue
            # DeltaParam parameters are typically named 'delta'
            if param.requires_grad and hasattr(m, 'delta'):
                 if param.shape == m.delta.shape: 
                     pass # Logic to ensure parameter is captured

    # Iterate through all named parameters to capture any remaining adapter weights
    # specifically looking for 'lora_' or 'delta' in the parameter name.
    for n, p in module.named_parameters():
        if p.requires_grad and (('lora_' in n) or ('delta' in n)):
            if p not in params:
                params.append(p)
    return params