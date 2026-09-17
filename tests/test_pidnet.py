"""Smoke tests for PIDNet-S integration."""

import sys
from pathlib import Path

import torch

# Ensure code/ is importable
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from models.baselines import PIDNetSBinaryWrapper
from models.factory import build_model, build_model_from_checkpoint


def test_pidnet_s_output_shape():
    """PIDNet-S binary wrapper must output (B, 1, H, W)."""
    model = PIDNetSBinaryWrapper()
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1, 256, 256), f"expected (2,1,256,256), got {y.shape}"
    assert torch.isfinite(y).all(), "output contains NaN/Inf"


def test_pidnet_s_param_count():
    """PIDNet-S should be ~7-8M parameters (official: ~7.6M)."""
    model = PIDNetSBinaryWrapper()
    n = sum(p.numel() for p in model.parameters())
    assert 6_000_000 < n < 9_000_000, (
        f"PIDNet-S param count {n:,} out of expected range (6M–9M)"
    )


def test_pidnet_s_factory_via_args():
    """build_model(fake_args) should return a PIDNetSBinaryWrapper."""

    class FakeArgs:
        model = "pidnet_s"
        pidnet_pretrained = ""
        encoder_output_stride = 8
        global_branch = "avg"
        phm_n = 2
        aspp_out_ch = 256
        decoder_ch = 256
        low_ch = 48
        rates = (12, 24, 36)
        sk_reduction = 16

    model = build_model(FakeArgs())
    assert isinstance(model, PIDNetSBinaryWrapper)

    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 128, 128)


def test_pidnet_s_checkpoint_roundtrip(tmp_path: Path):
    """Train a fake checkpoint, then reload via factory."""
    model = PIDNetSBinaryWrapper()
    model.eval()

    ckpt = {
        "model": model.state_dict(),
        "epoch": 1,
        "score": 0.5,
        "model_name": "pidnet_s",
        "seed": 42,
        "encoder_output_stride": 8,
    }
    ckpt_path = tmp_path / "fake_pidnet.pt"
    torch.save(ckpt, ckpt_path)

    loaded, loaded_ckpt = build_model_from_checkpoint(ckpt_path, torch.device("cpu"))
    assert isinstance(loaded, PIDNetSBinaryWrapper)
    assert loaded_ckpt["model_name"] == "pidnet_s"

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = loaded(x)
    assert y.shape == (1, 1, 128, 128)
