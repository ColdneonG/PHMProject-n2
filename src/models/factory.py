"""Centralized model factory shared by train / eval / predict."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .baselines import build_baseline_model
from .phm_project import (
    AblationDeepLabV3Plus,
    LocalSKGlobalBypassDeepLabV3Plus,
    StrictSupplementaryDeepLabV3Plus,
    _MODEL_REGISTRY,
)


def get_effective_model_config(args) -> dict | None:
    """Resolve the complete construction config used for this invocation."""
    reg = _MODEL_REGISTRY.get(args.model)
    if reg is None:
        return None

    config = dict(reg)
    config.update(
        {
            "model_key": args.model,
            "global_branch": args.global_branch,
            "encoder_output_stride": int(args.encoder_output_stride),
            "aspp_out_ch": int(args.aspp_out_ch),
            "decoder_ch": int(args.decoder_ch),
            "low_ch": int(args.low_ch),
            "rates": [int(x) for x in args.rates],
            "sk_reduction": int(args.sk_reduction),
        }
    )

    named_n = reg.get("phm_n")
    if reg.get("projection_type") == "phm":
        if named_n is not None and int(args.phm_n) != int(named_n):
            raise ValueError(
                f"model key {args.model!r} requires --phm_n {named_n}, "
                f"but received {args.phm_n}"
            )
        config["phm_n"] = int(named_n if named_n is not None else args.phm_n)
    else:
        config["phm_n"] = None

    if len(config["rates"]) != 3:
        raise ValueError("--rates must contain exactly three values")
    return config


def _build_registered_model(
    model_name: str, config: dict, encoder_weights: str | None
) -> nn.Module:
    cls = config["cls"]
    if cls == "original":
        model = LocalSKGlobalBypassDeepLabV3Plus(
            global_branch=config["global_branch"],
            encoder_output_stride=config["encoder_output_stride"],
            aspp_out_ch=config["aspp_out_ch"],
            decoder_ch=config["decoder_ch"],
            low_ch=config["low_ch"],
            rates=tuple(config["rates"]),
            sk_reduction=config["sk_reduction"],
            phm_n=config["phm_n"],
            encoder_weights=encoder_weights,
        )
    elif cls == "ablation":
        model = AblationDeepLabV3Plus(
            branch_type=config["branch_type"],
            use_global_branch=config["use_global_branch"],
            global_branch=config["global_branch"],
            projection_type=config["projection_type"],
            phm_n=config["phm_n"] or 2,
            encoder_output_stride=config["encoder_output_stride"],
            aspp_out_ch=config["aspp_out_ch"],
            decoder_ch=config["decoder_ch"],
            low_ch=config["low_ch"],
            rates=tuple(config["rates"]),
            sk_reduction=config["sk_reduction"],
            dropout=config["dropout"],
            encoder_weights=encoder_weights,
        )
    elif cls == "supplementary":
        model = StrictSupplementaryDeepLabV3Plus(
            config=config, encoder_weights=encoder_weights
        )
    else:
        raise ValueError(f"unknown registered model class {cls!r} for {model_name!r}")

    model._model_config = dict(config)
    return model


def build_model(args, encoder_weights_override="__registry__") -> nn.Module:
    """Build a model from command-line arguments (used by train.py)."""
    config = get_effective_model_config(args)
    if config is not None:
        encoder_weights = (
            config["encoder_weights"]
            if encoder_weights_override == "__registry__"
            else encoder_weights_override
        )
        return _build_registered_model(
            args.model, config, encoder_weights=encoder_weights
        )

    return build_baseline_model(
        args.model,
        encoder_output_stride=getattr(args, "encoder_output_stride", 8),
        pidnet_pretrained=getattr(args, "pidnet_pretrained", ""),
    )


def _legacy_config(model_name: str, ckpt: dict) -> dict:
    """Reconstruct registered checkpoints that predate full model_config."""
    config = dict(_MODEL_REGISTRY[model_name])
    config.update(
        {
            "model_key": model_name,
            "global_branch": ckpt.get("global_branch", config["global_branch"]),
            "encoder_output_stride": int(
                ckpt.get("encoder_output_stride", config["encoder_output_stride"])
            ),
            "rates": list(ckpt.get("rates", config["rates"])),
            "phm_n": ckpt.get("phm_n", config["phm_n"]),
        }
    )
    if config["projection_type"] == "phm" and config["phm_n"] is None:
        config["phm_n"] = 2
    return config


def build_model_from_checkpoint(
    ckpt_path: Path, device: torch.device
) -> tuple[nn.Module, dict]:
    """Build strictly from saved full config, with legacy checkpoint fallback."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        model = LocalSKGlobalBypassDeepLabV3Plus(encoder_weights=None)
        model.load_state_dict(ckpt, strict=True)
        model.to(device).eval()
        return model, ckpt

    model_name = ckpt.get("model_name", "phm_project_n2")
    if model_name in _MODEL_REGISTRY:
        saved_config = ckpt.get("model_config")
        if saved_config is not None:
            config = dict(saved_config)
            if config.get("model_key", model_name) != model_name:
                raise ValueError(
                    "checkpoint model_name/model_config.model_key mismatch: "
                    f"{model_name!r} vs {config.get('model_key')!r}"
                )
        else:
            config = _legacy_config(model_name, ckpt)

        named_n = _MODEL_REGISTRY[model_name].get("phm_n")
        if named_n is not None and int(config.get("phm_n")) != int(named_n):
            raise ValueError(
                f"checkpoint config conflicts with model key {model_name!r}: "
                f"phm_n={config.get('phm_n')} expected {named_n}"
            )
        model = _build_registered_model(model_name, config, encoder_weights=None)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model, ckpt

    model = build_baseline_model(
        model_name,
        encoder_output_stride=ckpt.get("encoder_output_stride", 8),
        pidnet_pretrained="",
    )
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, ckpt
