from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerBinaryWrapper(nn.Module):
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class PIDNetSBinaryWrapper(nn.Module):
    """PIDNet-S adapted to the project's binary-logit interface."""

    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()

        from .pidnet_official.pidnet import get_pred_model

        # augment=False：只返回主分割头
        # num_classes=1：适配 BCEWithLogitsLoss 二值分割
        self.model = get_pred_model(
            name="pidnet_s",
            num_classes=1,
        )

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str) -> None:
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )

        # 官方 PIDNet ImageNet checkpoint 格式: {"state_dict": {...}, ...}
        state = checkpoint.get("state_dict", checkpoint)
        model_state = self.model.state_dict()

        compatible = {}
        skipped_shape: dict[str, str] = {}
        for key, value in state.items():
            # 兼容 module.xxx、model.xxx 等前缀
            candidates = [
                key,
                key.removeprefix("module."),
                key.removeprefix("model."),
                key.removeprefix("module.model."),
            ]

            matched = False
            for candidate in candidates:
                if candidate in model_state:
                    if model_state[candidate].shape == value.shape:
                        compatible[candidate] = value
                        matched = True
                        break
                    else:
                        skipped_shape[candidate] = (
                            f"ckpt={tuple(value.shape)}"
                            f" vs model={tuple(model_state[candidate].shape)}"
                        )

            # 不在 model_state 中的 key 直接忽略（分类头等）

        result = self.model.load_state_dict(compatible, strict=False)

        # ---- detailed diagnostics ----
        total_model = len(model_state)
        loaded = len(compatible)
        pct = 100 * loaded / max(total_model, 1)
        print(
            f"[INFO] PIDNet-S pretrained: {loaded}/{total_model} "
            f"tensors ({pct:.1f}%) from {path}"
        )
        if skipped_shape:
            shape_info = "; ".join(
                f"{k}: {v}" for k, v in list(skipped_shape.items())[:5]
            )
            print(
                f"[INFO]   shape-mismatch (skipped): "
                f"{len(skipped_shape)} tensors, e.g. {shape_info}"
            )
        print(
            f"[INFO]   strict missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)}"
        )
        if loaded < total_model * 0.3:
            print(
                "[WARN]   <30% weights loaded! Check pretrained path and model variant."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)

        # PIDNet 主输出约为输入的 1/8 分辨率
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


def build_baseline_model(
    model_name: str,
    encoder_output_stride: int = 8,
    pidnet_pretrained: str = "",
) -> nn.Module:
    if model_name == "pidnet_s":
        return PIDNetSBinaryWrapper(
            pretrained_path=pidnet_pretrained or None,
        )

    import segmentation_models_pytorch as smp

    if model_name == "unet_resnet18_pretrained":
        return smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    if model_name == "deeplabv3plus_resnet18_pretrained":
        try:
            return smp.DeepLabV3Plus(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                encoder_output_stride=encoder_output_stride,
                in_channels=3,
                classes=1,
                activation=None,
            )
        except Exception as exc:
            print(
                f"[WARN] DeepLabV3+ output_stride={encoder_output_stride} failed; fallback output_stride=16. error={exc}"
            )
            return smp.DeepLabV3Plus(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                encoder_output_stride=16,
                in_channels=3,
                classes=1,
                activation=None,
            )
    if model_name == "pspnet_resnet18_pretrained":
        try:
            return smp.PSPNet(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=None,
            )
        except TypeError:
            return smp.PSPNet(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                classes=1,
                activation=None,
            )
    if model_name == "segformer_b0_pretrained":
        return SegFormerBinaryWrapper()
    raise ValueError(f"unknown baseline model: {model_name}")
