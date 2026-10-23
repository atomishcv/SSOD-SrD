"""
Module 2: Query-Norm Objectness Adapter (QNorm-Obj)
====================================================
Paper: EW-DETR, Section 3.4

Motivation
----------
In DETR-based detectors, decoder queries matched to real objects empirically
develop larger L2 norms than unmatched (background) queries.  QNorm-Obj exploits
this to produce two decoupled signals from the same decoder feature h_i:

  1. **Semantic direction** (norm-free, domain-invariant):
         h_norm = LN(h_i) / ||LN(h_i)||_2       ← lives on the unit sphere

  2. **Objectness cue** (magnitude-based, class-agnostic):
         z_obj  = f_obj(||h_i||_2) / (τ + ε)    ← scalar logit

Classification feature is a learnable convex mix:
    h_cls = (1 - α_mix) * h_i + α_mix * h_norm   (Eq. 8)

Classification logits:
    z_cls = W_cls @ h_cls + b_cls                 (Eq. 9)

Objectness logit (passed to EUMix / UnseenPhenotype):
    z_obj = f_obj(||h_i||_2) / (τ + ε)           (Eq. 10)

Key properties
--------------
- **No auxiliary loss**: both heads are trained implicitly through the standard
  detection loss.  Matched queries get positive class + box gradients, which
  encourages their norms and directions to jointly encode objectness + semantics.
- **Domain invariance**: because h_norm lives on the unit sphere, domain-induced
  covariate shifts that manifest as global amplitude changes are cancelled.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNormObjectnessAdapter(nn.Module):
    """
    Query-Norm Objectness Adapter.

    Parameters
    ----------
    embed_dim : int
        Dimension of the DETR decoder output features (e.g. 256 for RF-DETR).
    num_known_classes : int
        Number of known classes at the *current* task (|K_t|).
        An additional "unknown" logit is appended, so the final classification
        head output has shape (..., num_known_classes + 1).
    obj_hidden_dim : int
        Hidden size of the objectness MLP.
    temperature : float  (τ)
        Temperature scaling denominator for the objectness logit.
    eps : float          (ε)
        Numerical stability constant.
    """

    def __init__(
        self,
        embed_dim: int,
        num_known_classes: int,
        obj_hidden_dim: int = 64,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_known_classes = num_known_classes
        self.temperature = temperature
        self.eps = eps

        # LayerNorm applied before L2 normalisation (Eq. 7)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Learnable mixing coefficient α_mix ∈ (0, 1)  (Eq. 8)
        # Initialised to 0.5 (equal mix of raw and normalised)
        self._alpha_mix_logit = nn.Parameter(torch.zeros(1))

        # Classification head: maps h_cls → [z_known (|K_t|), z_unknown (1)]  (Eq. 9)
        self.cls_head = nn.Linear(embed_dim, num_known_classes + 1)

        # Objectness MLP: f_obj: scalar → scalar  (Eq. 10)
        self.obj_head = nn.Sequential(
            nn.Linear(1, obj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obj_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha_mix(self) -> torch.Tensor:
        """α_mix ∈ (0, 1), learned."""
        return torch.sigmoid(self._alpha_mix_logit)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, h: torch.Tensor):
        """
        Parameters
        ----------
        h : Tensor, shape (..., embed_dim)
            Raw decoder output features for a batch of queries.

        Returns
        -------
        z_cls  : Tensor (..., num_known_classes + 1)
            Classification logits (known classes + unknown).
        z_obj  : Tensor (..., 1)
            Objectness logit (class-agnostic; used by EUMix / UnseenPhenotype).
        h_cls  : Tensor (..., embed_dim)
            The mixed classification feature (useful for downstream inspection).
        """
        # Step 1: compute norm (objectness cue) BEFORE LayerNorm  (Eq. 10)
        norm_h = h.norm(dim=-1, keepdim=True)                   # (..., 1)

        # Step 2: direction-only vector on unit sphere  (Eq. 7)
        h_ln = self.layer_norm(h)                               # (..., D)
        h_norm = h_ln / (h_ln.norm(dim=-1, keepdim=True) + self.eps)  # (..., D)

        # Step 3: learnable convex combination  (Eq. 8)
        alpha = self.alpha_mix                                  # scalar
        h_cls = (1.0 - alpha) * h + alpha * h_norm             # (..., D)

        # Step 4: classification logits  (Eq. 9)
        z_cls = self.cls_head(h_cls)                            # (..., |K_t|+1)

        # Step 5: objectness logit  (Eq. 10)
        z_obj = self.obj_head(norm_h) / (self.temperature + self.eps)   # (..., 1)

        return z_cls, z_obj, h_cls

    # ------------------------------------------------------------------
    # Utility: update head for new task (grow the classification head)
    # ------------------------------------------------------------------

    def extend_classes(self, num_new_classes: int):
        """
        Extend the classification head to accommodate new classes introduced
        at the next task.  Old weights are preserved; new weights are zero-init.

        Parameters
        ----------
        num_new_classes : int  (|K_{t+1}| - |K_t|)
        """
        old_weight = self.cls_head.weight.data    # (|K_t|+1, D)
        old_bias   = self.cls_head.bias.data      # (|K_t|+1,)
        old_n = self.num_known_classes

        new_n = old_n + num_new_classes
        new_head = nn.Linear(self.embed_dim, new_n + 1).to(old_weight.device)
        with torch.no_grad():
            # Copy known-class weights
            new_head.weight[:old_n] = old_weight[:old_n]
            new_head.bias[:old_n]   = old_bias[:old_n]
            # Zero-init new known classes
            new_head.weight[old_n:new_n] = 0.0
            new_head.bias[old_n:new_n]   = 0.0
            # Unknown logit stays at the end
            new_head.weight[new_n] = old_weight[old_n]
            new_head.bias[new_n]   = old_bias[old_n]

        self.cls_head = new_head
        self.num_known_classes = new_n


# ---------------------------------------------------------------------------
# Convenience: patch QNorm-Obj onto a pre-built DETR model
# ---------------------------------------------------------------------------

class QNormObjWrapper(nn.Module):
    """
    Wraps an existing DETR-style detector so that its decoder output features
    are processed by QNormObjectnessAdapter instead of the original cls head.

    Typical usage::

        wrapper = QNormObjWrapper(detr_model, embed_dim=256, num_known_classes=10)
        # During training:
        z_cls, z_obj, _ = wrapper(images, targets)

    Parameters
    ----------
    detector : nn.Module
        Pre-built DETR-based detector.  Must expose ``get_decoder_features()``
        which returns the last-layer decoder output tensor of shape
        (batch, num_queries, embed_dim).  (Adapt to your specific DETR variant.)
    embed_dim : int
    num_known_classes : int
    kwargs : forwarded to QNormObjectnessAdapter
    """

    def __init__(
        self,
        detector: nn.Module,
        embed_dim: int,
        num_known_classes: int,
        **kwargs,
    ):
        super().__init__()
        self.detector = detector
        self.qnorm_obj = QNormObjectnessAdapter(
            embed_dim=embed_dim,
            num_known_classes=num_known_classes,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        # Run the backbone + encoder + decoder; extract query features
        hs = self.detector.get_decoder_features(*args, **kwargs)  # (B, Q, D)
        z_cls, z_obj, h_cls = self.qnorm_obj(hs)
        return z_cls, z_obj, h_cls
