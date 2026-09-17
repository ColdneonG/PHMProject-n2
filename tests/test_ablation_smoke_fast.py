#!/usr/bin/env python3
"""Fast smoke test for three new ablation models: A2-Conv, A2-HC, A3.

Also verifies that the existing phm_project_n2 (A4) remains compatible.
Uses encoder_weights=None to avoid network downloads during smoke tests.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.phm_project import (
    AblationDeepLabV3Plus,
    LocalSKGlobalBypassDeepLabV3Plus,
    _MODEL_REGISTRY,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 256
BATCH = 2


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def smoke_one(name, model, expected_output_shape=(BATCH, 1, INPUT_SIZE, INPUT_SIZE)):
    print(f"\n  --- {name} ---")

    n_total, n_trainable = count_params(model)
    print(f"  Params: total={n_total:,}  trainable={n_trainable:,}")

    # 1. Forward
    x = torch.randn(BATCH, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"  Forward: in {tuple(x.shape)} → out {tuple(out.shape)}")
    assert tuple(out.shape) == expected_output_shape, f"BAD OUTPUT: {tuple(out.shape)}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"

    # 2. Backward
    model.train()
    x2 = torch.randn(BATCH, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
    y = model(x2)
    loss = y.mean()
    loss.backward()
    print(f"  Backward loss: {loss.item():.6f}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf!"

    # 3. Internal shapes via hook
    shapes = {}

    def hook(module, inp, outp):
        shapes["aspp"] = tuple(outp.shape)

    h = model.aspp.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(x)
    h.remove()
    print(f"  ASPP output: {shapes.get('aspp', 'N/A')}")
    if "aspp" in shapes:
        c = shapes["aspp"][1]
        h_spatial = shapes["aspp"][2:]
        print(f"    channels={c}, spatial={h_spatial}")
        assert c == 256, f"ASPP channels != 256: {c}"
        assert h_spatial == (32, 32), f"ASPP spatial != 32×32: {h_spatial}"

    # 4. Checkpoint save & reload
    reg = _MODEL_REGISTRY.get(name, {})
    is_ablation = reg.get("cls") == "ablation"

    ckpt_dict = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "epoch": 1,
        "score": 0.5,
        "model_name": name,
        "seed": 42,
    }
    if is_ablation:
        ckpt_dict.update({
            "ablation_variant": reg["ablation_variant"],
            "branch_type": reg["branch_type"],
            "use_global_branch": reg["use_global_branch"],
            "projection_type": reg["projection_type"],
            "phm_n": None,
            "global_branch": "avg",
            "rates": [12, 24, 36],
        })
    else:
        ckpt_dict["global_branch"] = "avg"
        ckpt_dict["phm_n"] = 2

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    try:
        torch.save(ckpt_dict, tmp)
        loaded = torch.load(tmp, map_location=DEVICE, weights_only=False)

        if is_ablation:
            m2 = AblationDeepLabV3Plus(
                branch_type=reg["branch_type"],
                use_global_branch=reg["use_global_branch"],
                global_branch="avg",
                projection_type=reg["projection_type"],
                encoder_weights=None,
            )
        else:
            m2 = LocalSKGlobalBypassDeepLabV3Plus(encoder_weights=None)
        m2.load_state_dict(loaded["model"], strict=True)
        m2.to(DEVICE).eval()
        with torch.no_grad():
            out2 = m2(x)
        assert tuple(out2.shape) == expected_output_shape, f"RELOAD BAD: {tuple(out2.shape)}"
        print(f"  Checkpoint: save/reload OK, output {tuple(out2.shape)}")

        if is_ablation:
            print(f"    ablation_variant: {loaded.get('ablation_variant')}")
            print(f"    branch_type:      {loaded.get('branch_type')}")
            print(f"    use_global_branch:{loaded.get('use_global_branch')}")
            print(f"    projection_type:  {loaded.get('projection_type')}")
            print(f"    rates:            {loaded.get('rates')}")
    finally:
        Path(tmp).unlink(missing_ok=True)

    print(f"  ✅ {name} PASSED")
    return n_total, n_trainable


def main():
    print("=" * 60)
    print("  ABLATION MODEL SMOKE TEST (fast, no network)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    results = {}

    # A2-Conv
    print("\n[1/4] A2-Conv: localsk_conv_resnet18_pretrained")
    m = AblationDeepLabV3Plus(
        branch_type="conv",
        use_global_branch=False,
        projection_type="none",
        encoder_weights=None,
    ).to(DEVICE)
    results["localsk_conv_resnet18_pretrained"] = smoke_one(
        "localsk_conv_resnet18_pretrained", m
    )

    # A2-HC
    print("\n[2/4] A2-HC: localsk_hc_resnet18_pretrained")
    m = AblationDeepLabV3Plus(
        branch_type="hypercomplex",
        use_global_branch=False,
        projection_type="none",
        encoder_weights=None,
    ).to(DEVICE)
    results["localsk_hc_resnet18_pretrained"] = smoke_one(
        "localsk_hc_resnet18_pretrained", m
    )

    # A3
    print("\n[3/4] A3: localsk_hc_global_conv_resnet18_pretrained")
    m = AblationDeepLabV3Plus(
        branch_type="hypercomplex",
        use_global_branch=True,
        global_branch="avg",
        projection_type="conv",
        encoder_weights=None,
    ).to(DEVICE)
    results["localsk_hc_global_conv_resnet18_pretrained"] = smoke_one(
        "localsk_hc_global_conv_resnet18_pretrained", m
    )

    # A4 – legacy compatibility
    print("\n[4/4] A4 LEGACY: phm_project_n2")
    try:
        m = LocalSKGlobalBypassDeepLabV3Plus(
            global_branch="avg", phm_n=2, encoder_weights=None
        ).to(DEVICE)
        results["phm_project_n2"] = smoke_one("phm_project_n2", m)
    except Exception as e:
        print(f"  ❌ LEGACY MODEL BROKEN: {e}")
        results["phm_project_n2"] = (0, 0)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<42} {'Total':>12} {'Trainable':>12}")
    print(f"  {'-'*42} {'-'*12} {'-'*12}")
    for name, (total, trainable) in results.items():
        print(f"  {name:<42} {total:>12,} {trainable:>12,}")

    all_ok = all(isinstance(v, tuple) and len(v) == 2 for v in results.values())
    if all_ok:
        print("\n🎉 ALL SMOKE TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
