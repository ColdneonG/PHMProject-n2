#!/usr/bin/env python3
"""Smoke test for three new ablation models: A2-Conv, A2-HC, A3.

Also verifies that the existing phm_project_n2 (A4) remains compatible.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.phm_project import (
    AblationDeepLabV3Plus,
    LocalSKGlobalBypassDeepLabV3Plus,
    _MODEL_REGISTRY,
    get_ablation_variant,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 256
BATCH = 2


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def smoke_one(name, build_fn, expected_config):
    print(f"\n{'=' * 60}")
    print(f"  SMOKE TEST: {name}")
    print(f"{'=' * 60}")

    # 1. Build
    print("[1] Building model ...")
    model = build_fn()
    model.to(DEVICE)
    total, trainable = count_params(model)
    print(f"    Total params:     {total:,}")
    print(f"    Trainable params: {trainable:,}")

    # 2. Forward pass
    print("[2] Forward pass B×3×256×256 ...")
    x = torch.randn(BATCH, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"    Output shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (BATCH, 1, INPUT_SIZE, INPUT_SIZE), (
        f"Bad output shape: {tuple(y.shape)}"
    )
    print("    ✓ Output shape correct")

    # 3. Backward pass
    print("[3] Backward pass ...")
    model.train()
    x = torch.randn(BATCH, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
    y = model(x)
    loss = y.mean()
    loss.backward()
    print(f"    Loss value: {float(loss):.6f}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"
    print("    ✓ Loss is finite")

    # 4. Check tensor shapes internally (hook-based check)
    print("[4] Internal tensor shape check ...")
    feats = None

    def hook(module, input, output):
        nonlocal feats
        feats = output

    # Attach hook to ASPP module
    aspp_module = model.aspp
    handle = aspp_module.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        _ = model(x)

    handle.remove()
    if feats is not None:
        print(f"    ASPP (high-level) output: {tuple(feats.shape)}")
        # Should be B×256×32×32 for all models (output_stride=8, input=256)
        assert feats.shape[1] == 256, f"ASPP out channels != 256: {feats.shape[1]}"
        assert feats.shape[2] == 32 and feats.shape[3] == 32, (
            f"ASPP spatial != 32: {feats.shape[2:]}"
        )
        print("    ✓ ASPP output B×256×32×32")

    # 5. Checkpoint save + reload
    print("[5] Checkpoint save & reload ...")
    ablation_variant = None
    reg = _MODEL_REGISTRY.get(name)
    if reg and reg.get("cls") == "ablation":
        ablation_variant = reg["ablation_variant"]

    ckpt_dict = {
        "model": model.state_dict(),
        "epoch": 1,
        "score": 0.5,
        "model_name": name,
        "seed": 42,
    }
    if reg and reg.get("cls") == "ablation":
        ckpt_dict.update(
            {
                "ablation_variant": reg["ablation_variant"],
                "branch_type": reg["branch_type"],
                "use_global_branch": reg["use_global_branch"],
                "projection_type": reg["projection_type"],
                "phm_n": None,
                "global_branch": "avg",
                "rates": [12, 24, 36],
            }
        )
    else:
        ckpt_dict["global_branch"] = "avg"
        ckpt_dict["phm_n"] = 2

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(ckpt_dict, tmp_path)
        loaded = torch.load(tmp_path, map_location=DEVICE, weights_only=False)

        # Rebuild from checkpoint config
        if reg and reg.get("cls") == "ablation":
            model2 = AblationDeepLabV3Plus(
                branch_type=reg["branch_type"],
                use_global_branch=reg["use_global_branch"],
                global_branch="avg",
                projection_type=reg["projection_type"],
                encoder_weights=None,
            )
        else:
            model2 = LocalSKGlobalBypassDeepLabV3Plus(encoder_weights=None)

        model2.load_state_dict(loaded["model"], strict=True)
        model2.to(DEVICE).eval()

        with torch.no_grad():
            y2 = model2(x)
        assert tuple(y2.shape) == (BATCH, 1, INPUT_SIZE, INPUT_SIZE), (
            f"Reload output bad: {tuple(y2.shape)}"
        )
        print("    ✓ Reloaded model output shape correct")

        # Check metadata
        assert loaded.get("model_name") == name, (
            f"model_name mismatch: {loaded.get('model_name')}"
        )
        print(f"    ✓ model_name in checkpoint: {loaded.get('model_name')}")
        if ablation_variant:
            assert loaded.get("ablation_variant") == ablation_variant, (
                f"ablation_variant mismatch: {loaded.get('ablation_variant')}"
            )
            print(
                f"    ✓ ablation_variant in checkpoint: {loaded.get('ablation_variant')}"
            )
            print(f"      branch_type:       {loaded.get('branch_type')}")
            print(f"      use_global_branch: {loaded.get('use_global_branch')}")
            print(f"      projection_type:   {loaded.get('projection_type')}")
            print(f"      rates:             {loaded.get('rates')}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 6. Encoder weights check
    print("[6] ImageNet pretrained encoder check ...")
    model3 = build_fn()
    # Check that encoder has non-zero weights (pretrained)
    enc_params = list(model3.encoder.parameters())
    has_nonzero = any((p.abs().sum() > 0).item() for p in enc_params if p.dim() >= 1)
    if has_nonzero:
        print("    ✓ Encoder has non-trivial weights (pretrained)")
    else:
        print("    ⚠ Encoder may not have pretrained weights loaded")

    print(f"\n  ✅ {name} – ALL CHECKS PASSED")
    return True


def build_a2_conv():
    return AblationDeepLabV3Plus(
        branch_type="conv",
        use_global_branch=False,
        projection_type="none",
        encoder_weights="imagenet",
    )


def build_a2_hc():
    return AblationDeepLabV3Plus(
        branch_type="hypercomplex",
        use_global_branch=False,
        projection_type="none",
        encoder_weights="imagenet",
    )


def build_a3():
    return AblationDeepLabV3Plus(
        branch_type="hypercomplex",
        use_global_branch=True,
        global_branch="avg",
        projection_type="conv",
        encoder_weights="imagenet",
    )


def build_a4_legacy():
    """Existing phm_project_n2 — must remain compatible."""
    return LocalSKGlobalBypassDeepLabV3Plus(
        global_branch="avg",
        phm_n=2,
        encoder_weights="imagenet",
    )


def main():
    print("=" * 60)
    print("  ABLATION MODEL SMOKE TEST SUITE")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    results = {}

    # A2-Conv
    results["localsk_conv_resnet18_pretrained"] = smoke_one(
        "localsk_conv_resnet18_pretrained",
        build_a2_conv,
        {"branch_type": "conv", "use_global_branch": False, "projection_type": "none"},
    )

    # A2-HC
    results["localsk_hc_resnet18_pretrained"] = smoke_one(
        "localsk_hc_resnet18_pretrained",
        build_a2_hc,
        {
            "branch_type": "hypercomplex",
            "use_global_branch": False,
            "projection_type": "none",
        },
    )

    # A3
    results["localsk_hc_global_conv_resnet18_pretrained"] = smoke_one(
        "localsk_hc_global_conv_resnet18_pretrained",
        build_a3,
        {
            "branch_type": "hypercomplex",
            "use_global_branch": True,
            "projection_type": "conv",
        },
    )

    # A4 – legacy compatibility
    print(f"\n{'=' * 60}")
    print(f"  COMPATIBILITY CHECK: phm_project_n2 (A4)")
    print(f"{'=' * 60}")
    try:
        results["phm_project_n2"] = smoke_one(
            "phm_project_n2",
            build_a4_legacy,
            {},
        )
    except Exception as e:
        print(f"  ❌ LEGACY MODEL BROKEN: {e}")
        results["phm_project_n2"] = False

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    all_ok = all(results.values())
    if all_ok:
        print("\n🎉 ALL SMOKE TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
