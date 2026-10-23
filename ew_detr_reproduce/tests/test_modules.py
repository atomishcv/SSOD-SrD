"""
End-to-end tests for the three EW-DETR modules.

Run with:
    cd ew_detr_reproduce
    python -m pytest tests/test_modules.py -v
or simply:
    python tests/test_modules.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn


# ============================================================
# Module 1: Incremental Dual LoRA Adapters
# ============================================================

def test_lora_layer_trainable():
    from modules.incremental_lora import LoRALayer
    layer = LoRALayer(32, 64, rank=4, alpha=8.0, trainable=True)
    assert layer.A.requires_grad
    assert layer.B.requires_grad
    dW = layer.delta_weight()
    assert dW.shape == (64, 32)
    print("[PASS] LoRALayer (trainable): shape, requires_grad")


def test_lora_layer_frozen():
    from modules.incremental_lora import LoRALayer
    layer = LoRALayer(32, 64, rank=4, alpha=8.0, trainable=False)
    assert not layer.A.requires_grad
    assert not layer.B.requires_grad
    print("[PASS] LoRALayer (frozen): buffers, no grad")


def test_incremental_lora_linear_forward():
    from modules.incremental_lora import IncrementalLoRALinear
    lin = nn.Linear(64, 128)
    lora_lin = IncrementalLoRALinear.from_linear(lin, rank=8, lora_alpha=16.0)

    x = torch.randn(4, 64)
    out = lora_lin(x)
    assert out.shape == (4, 128), f"Expected (4,128), got {out.shape}"
    print("[PASS] IncrementalLoRALinear forward: output shape correct")


def test_merge_task_into_agg():
    from modules.incremental_lora import IncrementalLoRALinear, compute_beta
    lin = nn.Linear(32, 32)
    lora_lin = IncrementalLoRALinear.from_linear(lin, rank=4, lora_alpha=8.0)

    # Simulate forward + backward (just check merge doesn't crash)
    x = torch.randn(2, 32)
    out = lora_lin(x)
    out.sum().backward()

    beta_t = compute_beta(task_idx=2, n_current=500, n_prev_total=1000)
    assert 0.1 <= beta_t <= 0.9, f"beta_t={beta_t} out of range"

    delta_agg_before = lora_lin.agg.delta_weight().clone().detach()
    lora_lin.merge_task_into_agg(beta_t)
    delta_agg_after = lora_lin.agg.delta_weight().clone().detach()

    # Aggregate should have changed
    assert not torch.allclose(delta_agg_before, delta_agg_after, atol=1e-7), \
        "Aggregate adapter did not change after merge"
    # Task adapter should be near zero (re-initialised)
    assert torch.allclose(lora_lin.task.B, torch.zeros_like(lora_lin.task.B)), \
        "Task B not zeroed after merge"
    print(f"[PASS] merge_task_into_agg: beta_t={beta_t:.3f}, agg updated, task reset")


def test_compute_beta_boundary():
    from modules.incremental_lora import compute_beta
    b1 = compute_beta(1, 100, 0)
    assert b1 == 1.0, f"Task 1 beta should be 1.0, got {b1}"

    # Large current task → small beta (not too much influence)
    b_large = compute_beta(2, 10000, 100, beta_min=0.1, beta_max=0.9)
    assert b_large == 0.1, f"Expected beta_min=0.1, got {b_large}"

    # Small current task → large beta
    b_small = compute_beta(2, 1, 100, beta_min=0.1, beta_max=0.9)
    assert b_small <= 0.9
    print(f"[PASS] compute_beta: t1={b1}, large={b_large:.3f}, small={b_small:.3f}")


def test_attach_incremental_lora():
    from modules.incremental_lora import attach_incremental_lora, IncrementalLoRALinear
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    model = attach_incremental_lora(model, rank=4)
    lora_layers = [m for m in model.modules() if isinstance(m, IncrementalLoRALinear)]
    assert len(lora_layers) == 2, f"Expected 2 LoRA layers, got {len(lora_layers)}"
    print(f"[PASS] attach_incremental_lora: {len(lora_layers)} layers wrapped")


# ============================================================
# Module 2: QNorm Objectness Adapter
# ============================================================

def test_qnorm_obj_output_shapes():
    from modules.qnorm_obj import QNormObjectnessAdapter
    D, K, B, Q = 256, 10, 2, 100
    adapter = QNormObjectnessAdapter(embed_dim=D, num_known_classes=K)

    h = torch.randn(B, Q, D)
    z_cls, z_obj, h_cls = adapter(h)

    assert z_cls.shape == (B, Q, K + 1), f"z_cls shape: {z_cls.shape}"
    assert z_obj.shape == (B, Q, 1),     f"z_obj shape: {z_obj.shape}"
    assert h_cls.shape == (B, Q, D),     f"h_cls shape: {h_cls.shape}"
    print(f"[PASS] QNormObjectnessAdapter shapes: z_cls{z_cls.shape} z_obj{z_obj.shape}")


def test_qnorm_obj_unit_sphere():
    from modules.qnorm_obj import QNormObjectnessAdapter
    D, K = 64, 5
    adapter = QNormObjectnessAdapter(embed_dim=D, num_known_classes=K, eps=1e-8)

    h = torch.randn(10, D) * 100.0  # Very large features
    h_ln = adapter.layer_norm(h)
    h_norm = h_ln / (h_ln.norm(dim=-1, keepdim=True) + adapter.eps)

    assert torch.allclose(
        h_norm.norm(dim=-1),
        torch.ones(h_norm.shape[0]),
        atol=1e-5,
    )

    # Verify alpha_mix ∈ (0,1)
    alpha = adapter.alpha_mix.item()
    assert 0.0 < alpha < 1.0, f"alpha_mix={alpha} not in (0,1)"
    print(f"[PASS] QNormObjectnessAdapter alpha_mix={alpha:.4f} in (0,1)")


def test_qnorm_obj_extend_classes():
    from modules.qnorm_obj import QNormObjectnessAdapter
    D, K = 64, 10
    adapter = QNormObjectnessAdapter(embed_dim=D, num_known_classes=K)

    h = torch.randn(3, D)
    z_cls_before, _, _ = adapter(h)
    assert z_cls_before.shape[-1] == K + 1  # K known + 1 unknown

    adapter.extend_classes(5)
    z_cls_after, _, _ = adapter(h)
    assert z_cls_after.shape[-1] == K + 5 + 1, f"After extend: {z_cls_after.shape}"
    print(f"[PASS] extend_classes: {K}→{K+5} known, unknown preserved")


def test_qnorm_obj_gradients():
    from modules.qnorm_obj import QNormObjectnessAdapter
    D, K = 64, 5
    adapter = QNormObjectnessAdapter(embed_dim=D, num_known_classes=K)
    h = torch.randn(4, D, requires_grad=True)
    z_cls, z_obj, _ = adapter(h)
    loss = z_cls.sum() + z_obj.sum()
    loss.backward()
    assert h.grad is not None
    print("[PASS] QNormObjectnessAdapter: gradients flow correctly")


# ============================================================
# Module 3: Unseen Phenotype Discovery
# ============================================================

def test_unseen_phenotype_output_shape():
    from modules.unseen_phenotype import UnseenPhenotypeDiscovery
    B, Q, K = 2, 50, 8
    module = UnseenPhenotypeDiscovery()

    z_cls = torch.randn(B, Q, K + 1)   # K known + 1 unseen
    z_sal = torch.randn(B, Q, 1)

    z_final = module(z_cls, z_sal)
    assert z_final.shape == (B, Q, K + 1), f"z_final shape: {z_final.shape}"
    print(f"[PASS] UnseenPhenotypeDiscovery output shape: {z_final.shape}")


def test_unseen_phenotype_high_salience_uncertain():
    """
    A query with high salience AND high known-class entropy (uncertainty)
    should produce a higher unseen probability than a query with low salience.
    """
    from modules.unseen_phenotype import UnseenPhenotypeDiscovery
    K = 5
    module = UnseenPhenotypeDiscovery(init_b_unk=-3.0)  # suppress cls branch
    module.eval()

    # Query 1: high salience (z_sal >> 0), high uncertainty (uniform known logits)
    z_cls_uncertain = torch.zeros(1, K + 1)       # uniform → high entropy
    z_sal_high      = torch.tensor([[5.0]])        # high salience

    # Query 2: low salience, same uncertainty
    z_sal_low       = torch.tensor([[-5.0]])       # low salience

    z_out_high = module(z_cls_uncertain, z_sal_high)
    z_out_low  = module(z_cls_uncertain, z_sal_low)

    unk_high = torch.sigmoid(z_out_high[0, -1]).item()
    unk_low  = torch.sigmoid(z_out_low[0, -1]).item()

    assert unk_high > unk_low, (
        f"High-salience+uncertain should score higher than low-salience: "
        f"{unk_high:.4f} vs {unk_low:.4f}"
    )
    print(f"[PASS] Unseen prob high-sal={unk_high:.4f} > low-sal={unk_low:.4f}")


def test_unseen_phenotype_known_suppression():
    from modules.unseen_phenotype import UnseenPhenotypeDiscovery
    K = 4
    module = UnseenPhenotypeDiscovery(
        init_suppression_strength=1.0,
        learn_suppression=False,
    )
    module.eval()

    z_cls = torch.ones(1, K + 1) * 2.0
    z_sal = torch.tensor([[5.0]])  # high salience → strong suppression

    z_final = module(z_cls, z_sal)
    # Known logits should be suppressed below original 2.0
    assert (z_final[0, :-1] < 2.0).all(), "Known logits should be suppressed"
    print("[PASS] Soft suppression: known logits reduced for high-salience query")


def test_unseen_phenotype_learnable_suppression():
    from modules.unseen_phenotype import UnseenPhenotypeDiscovery
    module = UnseenPhenotypeDiscovery(init_suppression_strength=0.5)
    assert module.theta_lambda.requires_grad
    assert module.suppression_strength.item() > 0.0
    print("[PASS] Learnable suppression strength is positive and trainable")


def test_pheno_discovery_head_e2e():
    from modules.unseen_phenotype import PhenotypeDiscoveryHead
    D, K, B = 128, 7, 4
    head = PhenotypeDiscoveryHead(embed_dim=D, num_known_classes=K)

    h = torch.randn(B, D)
    z_final, z_sal, info = head(h)

    assert z_final.shape == (B, K + 1), f"z_final: {z_final.shape}"
    assert z_sal.shape   == (B, 1),     f"z_sal:   {z_sal.shape}"
    assert "z_cls_raw" in info
    print(f"[PASS] PhenotypeDiscoveryHead e2e: z_final{z_final.shape}")


# ============================================================
# Module 1 + 3: Standalone incremental phenotype model
# ============================================================

def test_standalone_ewpheno_incremental():
    from models.ew_detr import StandaloneEWPhenotypeModel

    model = StandaloneEWPhenotypeModel(
        feature_dim=128,
        num_known_classes=5,
        lora_rank=4,
        lora_alpha=8.0,
        hidden_dim=64,
    )

    # ── Task 1 ──
    x_t1 = torch.randn(32, 128)
    out_t1 = model(x_t1)
    assert out_t1["logits"].shape == (32, 6), f"Task1 logits: {out_t1['logits'].shape}"

    # Simulate training step
    loss = out_t1["logits"].sum() + out_t1["z_sal"].sum()
    loss.backward()

    # End of Task 1
    beta = model.on_task_end(n_samples_current=1000)
    print(f"  Task 1 → beta_t={beta:.3f}")

    # ── Task 2: 3 new phenotypes ──
    model.on_new_task(num_new_classes=3)
    x_t2 = torch.randn(16, 128)
    out_t2 = model(x_t2)
    assert out_t2["logits"].shape == (16, 9), f"Task2 logits: {out_t2['logits'].shape}"

    loss2 = out_t2["logits"].sum() + out_t2["z_sal"].sum()
    loss2.backward()

    beta2 = model.on_task_end(n_samples_current=200)
    print(f"  Task 2 → beta_t={beta2:.3f}")

    print("[PASS] StandaloneEWPhenotypeModel: 2-task incremental run successful")


def test_region_only_task_transition():
    from models.ew_detr import StandaloneEWPhenotypeModel

    model = StandaloneEWPhenotypeModel(
        feature_dim=32,
        num_known_classes=1,
        lora_rank=2,
        hidden_dim=16,
    )
    before = model.head.qnorm_obj.num_known_classes
    model.on_new_task()
    after = model.head.qnorm_obj.num_known_classes
    assert before == after == 1
    print("[PASS] Region-only task transition keeps class count unchanged")


# ============================================================
# Metrics
# ============================================================

def test_metrics():
    from utils.metrics import compute_all_scores
    scores = compute_all_scores(
        map_prev_per_task        = [50.0, 40.0],   # mAP on prev classes at final task
        map_curr_when_introduced = [60.0, 45.0],   # avg mAP when classes introduced
        u_recall_per_task        = [0.6, 0.7],
        wi_per_task              = [0.1, 0.05],
        a_ose_per_task           = [5.0, 3.0],
        gt_unk_per_task          = [50.0, 30.0],
        map_curr_generalisation  = [35.0, 28.0],
    )
    for k, v in scores.items():
        print(f"  {k}: {v}")
    assert scores["FSS"] > 0
    assert "FOGS" in scores
    print("[PASS] EWOD metrics computed successfully")


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    tests = [
        # Module 1
        test_lora_layer_trainable,
        test_lora_layer_frozen,
        test_incremental_lora_linear_forward,
        test_merge_task_into_agg,
        test_compute_beta_boundary,
        test_attach_incremental_lora,
        # Module 2
        test_qnorm_obj_output_shapes,
        test_qnorm_obj_unit_sphere,
        test_qnorm_obj_extend_classes,
        test_qnorm_obj_gradients,
        # Module 3
        test_unseen_phenotype_output_shape,
        test_unseen_phenotype_high_salience_uncertain,
        test_unseen_phenotype_known_suppression,
        test_unseen_phenotype_learnable_suppression,
        test_pheno_discovery_head_e2e,
        # Integrated
        test_standalone_ewpheno_incremental,
        test_region_only_task_transition,
        # Metrics
        test_metrics,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            print(f"\n── {test_fn.__name__}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
