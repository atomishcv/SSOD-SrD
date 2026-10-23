"""
Module 3: Unseen Phenotype Discovery Module
============================================
Adapted from: EW-DETR, Section 3.5 — Entropy-Aware Unknown Mixing (EUMix)
Paper reference: Appendix A for full mathematical formulation.

Background
----------
EUMix calibrates "unknown" detection by blending two independent signals:
  (1) objectness-driven unknown probability: high when there IS an object but
      none of the known classes explain it confidently (high entropy).
  (2) classifier-driven unknown probability: from a learned "unknown" logit.

Adaptation to Phenotype Discovery
-----------------------------------
In biological / clinical feature spaces (single-cell RNA, pathology patches,
multiomics embeddings, etc.), "object detection" becomes "phenotype slot
assignment": each query / feature vector is matched to one of the known
phenotypes, or flagged as a **novel (unseen) phenotype**.

We re-label the original components as follows:

  EUMix term            →  Phenotype-Discovery term
  ─────────────────────────────────────────────────
  objectness logit       →  phenotype-salience logit  (z_sal)
  unknown class          →  unseen-phenotype class
  known-class confidence →  known-phenotype confidence
  gap g_i                →  phenotype entropy gap

The mathematics is identical; only the semantic interpretation differs.
This makes the module applicable to:
  - Incremental cell-type discovery (scRNA-seq)
  - Novel lesion detection in pathology slides
  - Emerging cluster discovery in any embedding space

Mathematical formulation  (mirrors Appendix A of the paper)
-----------------------------------------------------------
Inputs per feature slot i:
    z_cls = [z_known (|K_t|), z_unk (1)]   ← known-phenotype + unseen logits
    z_sal                                   ← salience logit from QNormObj/custom

  (1) Max known-phenotype confidence:
        p_known_max = max_{c ∈ K_t}  σ(z_{i,c}^{known})           (Eq. 16)

  (2) Calibrated entropy gap:
        γ = softplus(θ_γ)                                           (temperature)
        g_i = (1 − p_known_max_i)^γ                                (Eq. 17)

  (3) Salience-driven unseen probability:
        p_sal_i^unseen = σ(z_sal_i) · g_i                          (Eq. 18)

  (4) Classifier-driven unseen probability (with learned bias):
        p_cls_i^unseen = σ(z_unk_i + b_unk)                        (Eq. 19)

  (5) Learnable mixture:
        α = σ(θ_α)                                                  (Eq. 20)
        p_final_i^unseen = α · p_cls_i + (1−α) · p_sal_i           (Eq. 21)

  (6) Convert back to logit:
        z_final_i^unseen = logit(p_final_i^unseen)                  (Eq. 22)

  (7) Soft suppression on known-phenotype logits:
        z_known_suppressed = z_known − λ · p_sal · g_i             (Eq. inline)

Final output:
    z_final = [z_known_suppressed, z_final_unseen]                  (Eq. 23)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnseenPhenotypeDiscovery(nn.Module):
    """
    Unseen Phenotype Discovery Module — adapted from EUMix.

    Can be used standalone on top of any feature encoder that produces:
      - classification logits z_cls  (... × (|K_t| + 1))
      - salience logit      z_sal   (... × 1)

    Parameters
    ----------
    init_alpha : float
        Initial value of the learnable mixing weight α (in logit space).
        Default 0.0 → α = sigmoid(0) = 0.5 (equal mix).
    init_gamma : float
        Initial value of θ_γ; γ = softplus(θ_γ) controls sharpness of the
        entropy gap.  Default 0.0 → γ ≈ 0.69 (smooth transition).
    init_b_unk : float
        Initial bias for the unknown logit to compensate for the lack of
        explicit unknown supervision.
    init_suppression_strength : float  (λ)
        Initial strength of the soft suppression applied to known-phenotype logits.
    learn_suppression : bool
        If True, learn λ jointly with the other EUMix parameters. If False,
        keep the supplied strength fixed.
    """

    def __init__(
        self,
        init_alpha: float = 0.0,
        init_gamma: float = 0.0,
        init_b_unk: float = -2.0,
        init_suppression_strength: float = 0.5,
        learn_suppression: bool = True,
    ):
        super().__init__()

        # θ_α: learnable mixing weight (α = sigmoid(θ_α))
        self.theta_alpha = nn.Parameter(torch.tensor(init_alpha))

        # θ_γ: learnable gap temperature (γ = softplus(θ_γ))
        self.theta_gamma = nn.Parameter(torch.tensor(init_gamma))

        # b_unk: learnable bias for the classifier-driven unknown branch
        self.b_unk = nn.Parameter(torch.tensor(init_b_unk))

        if init_suppression_strength < 0.0:
            raise ValueError("init_suppression_strength must be non-negative.")

        self.learn_suppression = learn_suppression
        init_lambda = torch.tensor(init_suppression_strength, dtype=torch.float32)
        if learn_suppression:
            raw_lambda = torch.log(torch.expm1(init_lambda.clamp_min(1e-6)))
            self.theta_lambda = nn.Parameter(raw_lambda)
        else:
            self.register_buffer("fixed_lambda", init_lambda)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        """Mixing weight α ∈ (0, 1); auto-balanced end-to-end."""
        return torch.sigmoid(self.theta_alpha)

    @property
    def gamma(self) -> torch.Tensor:
        """Temperature γ > 0 for the entropy gap."""
        return F.softplus(self.theta_gamma)

    @property
    def suppression_strength(self) -> torch.Tensor:
        """Positive soft-suppression scale λ."""
        if self.learn_suppression:
            return F.softplus(self.theta_lambda)
        return self.fixed_lambda

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_cls: torch.Tensor,
        z_sal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calibrate the unknown / unseen-phenotype logit.

        Parameters
        ----------
        z_cls : Tensor, shape (..., |K_t| + 1)
            Raw classification logits.  Last channel is the unknown logit z_unk.
        z_sal : Tensor, shape (..., 1)
            Salience logit from QNormObjectnessAdapter (or any salience model).

        Returns
        -------
        z_final : Tensor, shape (..., |K_t| + 1)
            Calibrated logits with a replaced unknown channel and soft-suppressed
            known channels.
        """
        z_known = z_cls[..., :-1]   # (..., |K_t|)
        z_unk   = z_cls[..., -1:]   # (..., 1)

        # ── Step 1: Max known-phenotype confidence  (Eq. 16) ──────────
        p_known_max = torch.sigmoid(z_known).max(dim=-1, keepdim=True).values  # (...,1)

        # ── Step 2: Calibrated entropy gap  (Eq. 17) ──────────────────
        gamma = self.gamma                          # scalar > 0
        g = (1.0 - p_known_max).pow(gamma)         # (..., 1); large when uncertain

        # ── Step 3: Salience-driven unseen probability  (Eq. 18) ──────
        p_sal_unseen = torch.sigmoid(z_sal) * g    # (..., 1)

        # ── Step 4: Classifier-driven unseen probability  (Eq. 19) ───
        p_cls_unseen = torch.sigmoid(z_unk + self.b_unk)  # (..., 1)

        # ── Step 5: Learnable mixture  (Eq. 21) ───────────────────────
        alpha = self.alpha                                 # scalar ∈ (0,1)
        p_final_unseen = alpha * p_cls_unseen + (1.0 - alpha) * p_sal_unseen  # (...,1)

        # ── Step 6: Back to logit space  (Eq. 22) ─────────────────────
        p_final_unseen = p_final_unseen.clamp(1e-6, 1.0 - 1e-6)
        z_final_unk = torch.log(p_final_unseen / (1.0 - p_final_unseen))     # (..., 1)

        # ── Step 7: Soft suppression on known-phenotype logits ─────────
        lambda_supp = self.suppression_strength
        if torch.any(lambda_supp > 0.0):
            suppression = lambda_supp * p_sal_unseen * g        # (..., 1)
            z_known_out = z_known - suppression
        else:
            z_known_out = z_known

        # ── Combine  (Eq. 23) ─────────────────────────────────────────
        z_final = torch.cat([z_known_out, z_final_unk], dim=-1)  # (..., |K_t|+1)
        return z_final

    # ------------------------------------------------------------------
    # Auxiliary: expose calibrated probabilities for analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_unseen_scores(
        self,
        z_cls: torch.Tensor,
        z_sal: torch.Tensor,
    ) -> dict:
        """
        Returns a dict of intermediate signals (for visualisation / debugging).

        Returns
        -------
        dict with keys:
            'p_known_max'   : max known-phenotype confidence per slot
            'entropy_gap'   : g_i
            'p_sal_unseen'  : salience-driven unseen probability
            'p_cls_unseen'  : classifier-driven unseen probability
            'p_final_unseen': final blended unseen probability
            'alpha'         : current mixing weight
            'gamma'         : current gap temperature
        """
        z_known = z_cls[..., :-1]
        z_unk   = z_cls[..., -1:]

        p_known_max   = torch.sigmoid(z_known).max(dim=-1, keepdim=True).values
        g             = (1.0 - p_known_max).pow(self.gamma)
        p_sal_unseen  = torch.sigmoid(z_sal) * g
        p_cls_unseen  = torch.sigmoid(z_unk + self.b_unk)
        alpha         = self.alpha
        p_final       = alpha * p_cls_unseen + (1.0 - alpha) * p_sal_unseen

        return {
            "p_known_max":    p_known_max,
            "entropy_gap":    g,
            "p_sal_unseen":   p_sal_unseen,
            "p_cls_unseen":   p_cls_unseen,
            "p_final_unseen": p_final,
            "alpha":          alpha.item(),
            "gamma":          self.gamma.item(),
            "suppression_strength": self.suppression_strength.item(),
        }


# ---------------------------------------------------------------------------
# Integrated head: QNorm-Obj + UnseenPhenotypeDiscovery
# ---------------------------------------------------------------------------

class PhenotypeDiscoveryHead(nn.Module):
    """
    Full head that combines QNorm-Obj feature processing and Unseen Phenotype
    Discovery in a single nn.Module, ready to replace the standard DETR cls head.

    Forward::

        z_final, z_obj, info = head(h)
        # z_final : calibrated logits, shape (B, Q, |K_t|+1)
        # z_obj   : salience logits,   shape (B, Q, 1)

    Parameters
    ----------
    embed_dim          : Decoder feature dimension.
    num_known_classes  : |K_t| at the current task.
    obj_hidden_dim     : Hidden units in the salience MLP.
    temperature        : Temperature τ for salience scaling.
    eps                : Numerical stability ε.
    init_alpha         : Initial α logit for EUMix.
    init_gamma         : Initial γ logit for EUMix.
    init_b_unk         : Initial bias for unknown logit.
    init_suppression_strength: Initial λ for soft suppression of known logits.
    learn_suppression: Learn λ end-to-end instead of keeping it fixed.
    """

    def __init__(
        self,
        embed_dim: int,
        num_known_classes: int,
        obj_hidden_dim: int = 64,
        temperature: float = 1.0,
        eps: float = 1e-6,
        init_alpha: float = 0.0,
        init_gamma: float = 0.0,
        init_b_unk: float = -2.0,
        init_suppression_strength: float = 0.5,
        learn_suppression: bool = True,
    ):
        super().__init__()
        # Avoid circular import; import here
        from .qnorm_obj import QNormObjectnessAdapter

        self.qnorm_obj = QNormObjectnessAdapter(
            embed_dim=embed_dim,
            num_known_classes=num_known_classes,
            obj_hidden_dim=obj_hidden_dim,
            temperature=temperature,
            eps=eps,
        )
        self.unseen_pheno = UnseenPhenotypeDiscovery(
            init_alpha=init_alpha,
            init_gamma=init_gamma,
            init_b_unk=init_b_unk,
            init_suppression_strength=init_suppression_strength,
            learn_suppression=learn_suppression,
        )

    def forward(self, h: torch.Tensor):
        """
        Parameters
        ----------
        h : Tensor (..., embed_dim)  — raw decoder features.

        Returns
        -------
        z_final : Tensor (..., |K_t|+1)  — calibrated class + unseen logits.
        z_sal   : Tensor (..., 1)         — salience logit.
        info    : dict with intermediate tensors (for loss / logging).
        """
        z_cls, z_sal, h_cls = self.qnorm_obj(h)
        z_final = self.unseen_pheno(z_cls, z_sal)

        info = {"z_cls_raw": z_cls, "z_sal": z_sal, "h_cls": h_cls}
        return z_final, z_sal, info

    def extend_classes(self, num_new_classes: int):
        """Grow the classification head for the next task."""
        self.qnorm_obj.extend_classes(num_new_classes)
