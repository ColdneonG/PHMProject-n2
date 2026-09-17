#!/usr/bin/env python3
"""Strict preflight for the 11-configuration supplementary ablation matrix."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.factory import (  # noqa: E402
    _build_registered_model,
    build_model_from_checkpoint,
    get_effective_model_config,
)
from models.phm_project import (  # noqa: E402
    MatchedStandardASPP,
    SKFusion,
    StrictSupplementaryASPP,
    _MODEL_REGISTRY,
)


MATRIX_KEYS = (
    "matched_a0_standard_aspp_resnet18_pretrained",
    "localsk_hc_global_conv_resnet18_pretrained",
    "phm_project_n2",
    "localsk_hc_global_insk_conv_resnet18_pretrained",
    "concat_hc_global_conv_resnet18_pretrained",
    "mean_hc_global_conv_resnet18_pretrained",
    "sum_hc_global_conv_resnet18_pretrained",
    "localsk_hc_global_group2_resnet18_pretrained",
    "localsk_hc_global_lowrank85_resnet18_pretrained",
    "phm_project_n1",
    "phm_project_n4",
)


def make_args(model_key: str, phm_n: int | None = None):
    named_n = _MODEL_REGISTRY[model_key].get("phm_n")
    return SimpleNamespace(
        model=model_key,
        global_branch="avg",
        encoder_output_stride=8,
        aspp_out_ch=256,
        decoder_ch=256,
        low_ch=48,
        rates=[12, 24, 36],
        sk_reduction=16,
        phm_n=phm_n if phm_n is not None else (named_n or 2),
    )


def count_trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def projector_operator(model):
    if hasattr(model.aspp, "project_operator"):
        return model.aspp.project_operator
    return model.aspp.project[0]


def build_no_download(model_key: str):
    config = get_effective_model_config(make_args(model_key))
    return _build_registered_model(model_key, config, encoder_weights=None), config


def test_registry_is_complete_and_immutable():
    assert set(MATRIX_KEYS).issubset(_MODEL_REGISTRY)
    required = {
        "cls",
        "context_block",
        "branch_type",
        "local_fusion",
        "global_mode",
        "global_branch",
        "projection_type",
        "projection_groups",
        "projection_rank",
        "phm_n",
        "rates",
        "encoder_name",
        "encoder_weights",
        "encoder_output_stride",
        "aspp_out_ch",
        "decoder_ch",
        "low_ch",
        "sk_reduction",
        "dropout",
    }
    for key in MATRIX_KEYS:
        assert required.issubset(_MODEL_REGISTRY[key]), key
    try:
        _MODEL_REGISTRY["illegal"] = {}  # type: ignore[index]
        raise AssertionError("registry accepted mutation")
    except TypeError:
        pass


def test_named_phm_order_mismatch_is_rejected():
    for key, wrong_n in (("phm_project_n1", 2), ("phm_project_n2", 4), ("phm_project_n4", 2)):
        try:
            get_effective_model_config(make_args(key, wrong_n))
            raise AssertionError(f"{key} accepted conflicting n={wrong_n}")
        except ValueError:
            pass


def test_structural_controls_and_parameter_budgets():
    a0, _ = build_no_download("matched_a0_standard_aspp_resnet18_pretrained")
    assert isinstance(a0.aspp, MatchedStandardASPP)
    assert a0.aspp.project_operator.in_channels == 1280
    assert a0.aspp.project_operator.out_channels == 256

    ginsk, _ = build_no_download("localsk_hc_global_insk_conv_resnet18_pretrained")
    assert isinstance(ginsk.aspp, StrictSupplementaryASPP)
    assert isinstance(ginsk.aspp.sk, SKFusion)
    assert ginsk.aspp.sk.num_branches == 5

    concat, _ = build_no_download("concat_hc_global_conv_resnet18_pretrained")
    assert concat.aspp.local_concat.in_channels == 1024
    assert concat.aspp.local_concat.out_channels == 256
    assert concat.aspp.local_concat.bias is None

    a4, _ = build_no_download("phm_project_n2")
    group2, _ = build_no_download("localsk_hc_global_group2_resnet18_pretrained")
    lowrank, _ = build_no_download("localsk_hc_global_lowrank85_resnet18_pretrained")
    counts = {
        "phm_n2": count_trainable(projector_operator(a4)),
        "group2": count_trainable(projector_operator(group2)),
        "lowrank85": count_trainable(projector_operator(lowrank)),
    }
    assert counts == {"phm_n2": 65544, "group2": 65536, "lowrank85": 65280}
    assert abs(counts["group2"] - counts["phm_n2"]) / counts["phm_n2"] <= 0.01
    assert abs(counts["lowrank85"] - counts["phm_n2"]) / counts["phm_n2"] <= 0.01


def test_pairwise_configuration_boundaries():
    ignored = {"model_key", "cls", "ablation_variant"}

    def diffs(a, b):
        ca = get_effective_model_config(make_args(a))
        cb = get_effective_model_config(make_args(b))
        return {k for k in ca if k not in ignored and ca[k] != cb[k]}

    assert diffs(
        "localsk_hc_global_conv_resnet18_pretrained",
        "localsk_hc_global_insk_conv_resnet18_pretrained",
    ) == {"global_mode"}
    for key in (
        "concat_hc_global_conv_resnet18_pretrained",
        "mean_hc_global_conv_resnet18_pretrained",
        "sum_hc_global_conv_resnet18_pretrained",
    ):
        assert diffs("localsk_hc_global_conv_resnet18_pretrained", key) == {
            "local_fusion"
        }
    for key in (
        "localsk_hc_global_group2_resnet18_pretrained",
        "localsk_hc_global_lowrank85_resnet18_pretrained",
    ):
        assert diffs("localsk_hc_global_conv_resnet18_pretrained", key).issubset(
            {"projection_type", "projection_groups", "projection_rank"}
        )
    assert diffs("phm_project_n1", "phm_project_n4") == {"phm_n"}


def test_all_models_forward_and_checkpoint_roundtrip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in MATRIX_KEYS:
        model, config = build_no_download(key)
        model.to(device).eval()
        x = torch.randn(2, 3, 256, 256, device=device)
        with torch.no_grad():
            output = model(x)
        assert tuple(output.shape) == (2, 1, 256, 256), key
        assert torch.isfinite(output).all(), key

        checkpoint = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_name": key,
            "model_config": config,
            "seed": 42,
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = Path(f.name)
        try:
            torch.save(checkpoint, ckpt_path)
            restored, loaded = build_model_from_checkpoint(ckpt_path, device)
            assert loaded["model_config"] == config
            with torch.no_grad():
                restored_output = restored(x)
            assert tuple(restored_output.shape) == (2, 1, 256, 256), key
        finally:
            ckpt_path.unlink(missing_ok=True)
        del model, restored, x, output, restored_output
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def test_audited_anchor_parameter_counts():
    a3, _ = build_no_download("localsk_hc_global_conv_resnet18_pretrained")
    a4, _ = build_no_download("phm_project_n2")
    assert count_trainable(a3) == 14_678_945
    assert count_trainable(a4) == 14_613_417


if __name__ == "__main__":
    tests = [
        test_registry_is_complete_and_immutable,
        test_named_phm_order_mismatch_is_rejected,
        test_structural_controls_and_parameter_budgets,
        test_pairwise_configuration_boundaries,
        test_all_models_forward_and_checkpoint_roundtrip,
        test_audited_anchor_parameter_counts,
    ]
    for test in tests:
        print(f">>> {test.__name__}")
        test()
    print("ALL STRICT SUPPLEMENTARY ABLATION PREFLIGHT TESTS PASSED")
