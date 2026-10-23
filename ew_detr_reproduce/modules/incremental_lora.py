"""
Module 1: Incremental Dual LoRA Adapters
=========================================
Paper: EW-DETR, Section 3.3

Two co-existing low-rank adapters attached to every targeted linear layer:

  Aggregate LoRA (ΔW_agg):
    - Frozen during training; accumulates compressed knowledge from all prior tasks.
    - Updated only at task boundaries via Data-Aware Merging + truncated SVD.

  Task-Specific LoRA (ΔW_task):
    - Trainable; captures current-task class/domain updates.
    - Reset to zero at each task transition after being merged into ΔW_agg.

Forward pass:
    y = W_base @ x + ΔW_agg @ x + ΔW_task @ x

Data-Aware Merging coefficient (Eq. 3):
    β_t = 1                                          if t == 1
    β_t = β_max - (β_max - β_min) * N_t / N_{1:t-1}  if t >= 2

Dense merge (Eq. 4):
    ΔW_merged = (1 - β_t) * ΔW_agg + β_t * ΔW_task

Truncated SVD projection back to rank r (Eq. 5):
    ΔW_merged ≈ U_r Σ_r V_r^T
    B_agg_new = U_r Σ_r,  A_agg_new = V_r^T
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-rank factor storage
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """
    Lightweight container for a single LoRA low-rank pair (B, A).

    The effective weight update is:
        ΔW = scaling * B @ A   shape: (out_features, in_features)

    B: (out_features, rank)
    A: (rank,       in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 16.0,
        trainable: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        if trainable:
            self.A = nn.Parameter(torch.empty(rank, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            # Stored as buffers so they travel with state_dict but do not receive grads
            self.register_buffer("A", torch.zeros(rank, in_features))
            self.register_buffer("B", torch.zeros(out_features, rank))

    def delta_weight(self) -> torch.Tensor:
        """Returns the full (out, in) weight update matrix."""
        return self.scaling * (self.B @ self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the LoRA correction to the *pre-activation* (i.e. applied after base linear)."""
        return x @ self.delta_weight().T


# ---------------------------------------------------------------------------
# Wrapped linear layer
# ---------------------------------------------------------------------------

class IncrementalLoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that attaches Incremental Dual LoRA Adapters.

    Usage
    -----
    Replace any linear layer ``lin`` inside a DETR encoder/decoder with::

        new_layer = IncrementalLoRALinear.from_linear(lin, rank=8)

    Then call ``new_layer.task_parameters()`` to get only the trainable LoRA
    parameters for the optimizer, and ``new_layer.merge_task_into_agg(...)``
    at each task boundary.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Frozen base weight (never receives gradients)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )

        # Aggregate adapter (frozen buffer-based)
        self.agg = LoRALayer(in_features, out_features, rank, alpha=lora_alpha, trainable=False)

        # Task-specific adapter (trainable)
        self.task = LoRALayer(in_features, out_features, rank, alpha=lora_alpha, trainable=True)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "IncrementalLoRALinear":
        """Wrap an existing nn.Linear, copying its pretrained weights."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            lora_alpha=lora_alpha,
            dropout=dropout,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if layer.bias is not None and linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_drop = self.dropout(x)
        out = F.linear(x, self.weight, self.bias)
        out = out + x_drop @ self.agg.delta_weight().T      # frozen history
        out = out + x_drop @ self.task.delta_weight().T     # current-task update
        return out

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def task_parameters(self):
        """Return only trainable (task-specific) LoRA parameters."""
        return [self.task.A, self.task.B]

    def freeze_task(self):
        self.task.A.requires_grad_(False)
        self.task.B.requires_grad_(False)

    def unfreeze_task(self):
        self.task.A.requires_grad_(True)
        self.task.B.requires_grad_(True)

    # ------------------------------------------------------------------
    # Task-boundary operations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def merge_task_into_agg(
        self,
        beta_t: float,
    ):
        """
        Data-aware merge of the task adapter into the aggregate adapter,
        then re-project to rank r via truncated SVD, and reset the task adapter.

        Parameters
        ----------
        beta_t : float
            Merging coefficient for the current task (computed externally by
            ``compute_beta``).  Controls plasticity vs. stability tradeoff.
        """
        # Full-rank delta matrices
        dW_agg = self.agg.delta_weight()    # (out, in)
        dW_task = self.task.delta_weight()  # (out, in)

        # Dense merge  (Eq. 4)
        dW_merged = (1.0 - beta_t) * dW_agg + beta_t * dW_task

        # Truncated SVD  (Eq. 5)
        try:
            U, S, Vh = torch.linalg.svd(dW_merged, full_matrices=False)
        except RuntimeError:
            # Fallback for older PyTorch
            U, S, V = torch.svd(dW_merged)
            Vh = V.T

        r = min(self.rank, S.shape[0])
        U_r = U[:, :r]   # (out, r)
        S_r = S[:r]      # (r,)
        V_r = Vh[:r, :]  # (r, in)

        # Store as raw factors (scaling already encoded in S_r)
        # We set agg.scaling = 1 by absorbing it into B/A directly.
        # B_agg_new = U_r * S_r,  A_agg_new = V_r
        # Then agg.delta_weight() = agg.scaling * B @ A
        # To make it consistent with the stored scaling, divide by scaling:
        self.agg.B.copy_(U_r * S_r.unsqueeze(0) / self.agg.scaling)  # absorb S into B
        self.agg.A.copy_(V_r)

        # Reset task adapter  (Eq. 6 / ΔW_task → 0)
        nn.init.kaiming_uniform_(self.task.A, a=math.sqrt(5))
        nn.init.zeros_(self.task.B)


# ---------------------------------------------------------------------------
# Merging coefficient (data-aware)
# ---------------------------------------------------------------------------

def compute_beta(
    task_idx: int,
    n_current: int,
    n_prev_total: int,
    beta_min: float = 0.1,
    beta_max: float = 0.9,
) -> float:
    """
    Compute the data-aware merging coefficient β_t (Eq. 3).

    Parameters
    ----------
    task_idx   : 1-based task index (1 = first task).
    n_current  : number of training samples in the current task.
    n_prev_total : cumulative samples from all previous tasks.
    beta_min, beta_max : bounds for β_t.

    Returns
    -------
    float : β_t ∈ [beta_min, beta_max]
    """
    if task_idx == 1:
        return 1.0  # Initialise aggregate entirely from task 1

    if n_current < 0 or n_prev_total < 0:
        raise ValueError("Sample counts must be non-negative.")

    ratio = n_current / max(n_prev_total, 1)
    beta = beta_max - (beta_max - beta_min) * ratio
    return float(max(beta_min, min(beta_max, beta)))


# ---------------------------------------------------------------------------
# Model-level helper: wrap all linear layers in a DETR encoder/decoder
# ---------------------------------------------------------------------------

def attach_incremental_lora(
    model: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[tuple] = None,
) -> nn.Module:
    """
    Replace every nn.Linear inside *model* (or a subset matching target_modules
    name patterns) with IncrementalLoRALinear.

    Parameters
    ----------
    model          : DETR-based detector (encoder + decoder must be mutable).
    rank           : LoRA rank r.
    lora_alpha     : LoRA alpha (scaling = alpha / rank).
    dropout        : Dropout probability on the LoRA path.
    target_modules : Tuple of name substrings to match (e.g. ('q_proj','v_proj')).
                     If None, all nn.Linear layers are wrapped.

    Returns
    -------
    The same model with replaced layers (in-place modification).
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None and not any(t in name for t in target_modules):
            continue

        # Navigate to parent
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]

        new_layer = IncrementalLoRALinear.from_linear(
            module, rank=rank, lora_alpha=lora_alpha, dropout=dropout
        )
        setattr(parent, child_name, new_layer)

    return model


def collect_task_parameters(model: nn.Module) -> list:
    """Collect all trainable task-specific LoRA parameters from the model."""
    params = []
    for module in model.modules():
        if isinstance(module, IncrementalLoRALinear):
            params.extend(module.task_parameters())
    return params


def merge_all_task_adapters(
    model: nn.Module,
    task_idx: int,
    n_current: int,
    n_prev_total: int,
    beta_min: float = 0.1,
    beta_max: float = 0.9,
):
    """
    Call ``merge_task_into_agg`` on every IncrementalLoRALinear in the model.
    Should be called once at the end of each task, before moving to the next task.
    """
    beta_t = compute_beta(task_idx, n_current, n_prev_total, beta_min, beta_max)
    for module in model.modules():
        if isinstance(module, IncrementalLoRALinear):
            module.merge_task_into_agg(beta_t)
    return beta_t
