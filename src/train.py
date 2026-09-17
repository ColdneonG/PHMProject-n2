from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from framework.data import LiquidDataset, make_sampler
from framework.metrics import evaluate, save_detail_tables
from framework.training import train_one_epoch
from framework.utils import (
    choose_device,
    count_trainable_params,
    ensure_dir,
    generate_experiment_dir,
    set_seed,
    write_json,
)
from models.baselines import build_baseline_model
from models.factory import build_model, get_effective_model_config
from models.phm_project import (
    _MODEL_REGISTRY,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_identifier() -> str:
    """Stable digest of all Python sources used by this code snapshot."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _manifest_hashes(root: Path) -> dict[str, str]:
    names = (
        "split_manifest.csv",
        "condition_image_level.csv",
        "group_assignment.csv",
        "split_summary.csv",
        "no_leakage_report.json",
    )
    return {name: _sha256_file(root / name) for name in names if (root / name).is_file()}


def _environment_record() -> dict:
    try:
        import segmentation_models_pytorch as smp

        smp_version = smp.__version__
    except Exception:
        smp_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "segmentation_models_pytorch": smp_version,
    }


def _projector_trainable_params(model) -> int | None:
    aspp = getattr(model, "aspp", None)
    if aspp is None:
        return None
    operator = getattr(aspp, "project_operator", None)
    if operator is None:
        project = getattr(aspp, "project", None)
        if isinstance(project, torch.nn.Sequential) and len(project) > 0:
            operator = project[0]
    if operator is None:
        return None
    return count_trainable_params(operator)


def _build_run_metadata(args, model, root: Path, n_params: int) -> dict:
    run_script = Path(args.run_script).resolve() if args.run_script else None
    return {
        "source_sha256": _source_identifier(),
        "data_manifest_sha256": _manifest_hashes(root),
        "run_script": str(run_script) if run_script else None,
        "run_script_sha256": (
            _sha256_file(run_script) if run_script and run_script.is_file() else None
        ),
        "environment": _environment_record(),
        "model_config": dict(getattr(model, "_model_config", {})),
        "trainable_params": n_params,
        "projector_operator_trainable_params": _projector_trainable_params(model),
    }


def _make_checkpoint_dict(model, args, epoch, score):
    """Build a checkpoint dict with complete reproducibility metadata."""
    metadata = dict(args._run_metadata)
    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "score": score,
        "model_name": args.model,
        "seed": args.seed,
        "encoder_output_stride": args.encoder_output_stride,
        "model_config": metadata["model_config"],
        "run_metadata": metadata,
    }

    # pidnet pretrained path & SHA256 for reproducibility
    pidnet_path = getattr(args, "pidnet_pretrained", "")
    if pidnet_path:
        ckpt["pidnet_pretrained"] = pidnet_path
        try:
            ckpt["pidnet_pretrained_sha256"] = hashlib.sha256(
                Path(pidnet_path).read_bytes()
            ).hexdigest()
        except Exception:
            ckpt["pidnet_pretrained_sha256"] = None

    reg = _MODEL_REGISTRY.get(args.model)
    if reg is not None:
        config = metadata["model_config"]
        ckpt.update(
            {
                "ablation_variant": config.get("ablation_variant"),
                "branch_type": config.get("branch_type"),
                "use_global_branch": config.get("use_global_branch"),
                "local_fusion": config.get("local_fusion"),
                "global_mode": config.get("global_mode"),
                "projection_type": config.get("projection_type"),
                "projection_groups": config.get("projection_groups"),
                "projection_rank": config.get("projection_rank"),
                "phm_n": config.get("phm_n"),
                "global_branch": config.get("global_branch"),
                "rates": list(config.get("rates", args.rates)),
            }
        )

    return ckpt


def cmd_train_eval(args):
    set_seed(args.seed)
    root = Path(args.root)
    if args.dry_run:
        config = get_effective_model_config(args)
        # Parameter/config inspection does not need to download pretrained values;
        # the effective config still records encoder_weights='imagenet'.
        model = build_model(args, encoder_weights_override=None)
        n_params = count_trainable_params(model)
        payload = {
            "model": args.model,
            "seed": args.seed,
            "effective_config": config,
            "trainable_params": n_params,
            "projector_operator_trainable_params": _projector_trainable_params(model),
            "intended_output": str(
                Path(args.out_dir)
                / f"{args.exp_name or 'expNNN'}_{args.model}_seed{args.seed}_{{UTCtimestamp}}"
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    out_dir = generate_experiment_dir(
        args.out_dir, args.model, args.seed, args.exp_name
    )
    print(f"[INFO] experiment directory: {out_dir}")
    device = choose_device(args.device)
    condition_csv = (
        Path(args.condition_csv)
        if args.condition_csv
        else root / "condition_image_level.csv"
    )

    train_ds = LiquidDataset(
        root,
        "train",
        input_size=args.input_size,
        condition_csv=condition_csv,
        train=True,
    )
    val_ds = LiquidDataset(
        root,
        "val",
        input_size=args.input_size,
        condition_csv=condition_csv,
        train=False,
    )
    test_ds = LiquidDataset(
        root,
        "test",
        input_size=args.input_size,
        condition_csv=condition_csv,
        train=False,
    )
    print(
        "[INFO] train conditions:",
        dict(Counter([m["condition"] for m in train_ds.meta])),
    )
    print(
        "[INFO] val conditions:", dict(Counter([m["condition"] for m in val_ds.meta]))
    )
    print(
        "[INFO] test conditions:", dict(Counter([m["condition"] for m in test_ds.meta]))
    )

    sampler = make_sampler(
        train_ds,
        video_weight_power=args.video_weight_power,
        condition_weight_power=args.condition_weight_power,
        small_weight=args.small_weight,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args).to(device)
    n_params = count_trainable_params(model)
    args._run_metadata = _build_run_metadata(args, model, root, n_params)
    print(f"[INFO] model={args.model}, params={n_params:,}, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = torch.amp.GradScaler(
        device="cuda", enabled=args.amp and device.type == "cuda"
    )

    best_score = -1e9
    best_epoch = -1
    best_path = out_dir / "best_model.pt"
    logs = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader, optimizer, device, amp=args.amp, scaler=scaler
        )
        scheduler.step()
        val_metrics, _ = evaluate(
            model, val_loader, device, threshold=args.threshold, amp=args.amp
        )
        score = float(val_metrics["selection_score"])
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            torch.save(
                _make_checkpoint_dict(model, args, epoch, best_score),
                best_path,
            )
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "val_selection_score": score,
            "val_overall_dice": val_metrics["overall_dice"],
            "val_overall_iou": val_metrics["overall_iou"],
            "val_clear_dice": val_metrics["condition_clear_dice"],
            "val_blur_dice": val_metrics["condition_blur_dice"],
            "val_small_dice": val_metrics["small_dice"],
            "val_boundary_f1": val_metrics["boundary_f1"],
            "val_level_mae": val_metrics["level_mae"],
            "best_score": best_score,
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - t0,
        }
        logs.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} loss={loss:.5f} val_dice={row['val_overall_dice']:.4f} score={score:.4f} best={best_score:.4f}@{best_epoch}"
        )
        if args.patience > 0 and (epoch - best_epoch) >= args.patience:
            print(
                f"[EARLY STOP] no improvement for {args.patience} epochs; best={best_score:.4f}@{best_epoch}"
            )
            break

    pd.DataFrame(logs).to_csv(
        out_dir / "training_log.csv", index=False, encoding="utf-8-sig"
    )

    # 保存最后一轮权重
    last_path = out_dir / "last_model.pt"
    torch.save(
        _make_checkpoint_dict(model, args, epoch, score),
        last_path,
    )
    print(
        f"[INFO] saved last checkpoint: {last_path} (epoch={epoch}, score={score:.4f})"
    )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_metrics, val_records = evaluate(
        model, val_loader, device, threshold=args.threshold, amp=args.amp
    )
    test_metrics, test_records = evaluate(
        model, test_loader, device, threshold=args.threshold, amp=args.amp
    )
    for metrics in (val_metrics, test_metrics):
        metrics.update(
            {
                "model": args.model,
                "params": n_params,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "preprocessing": "edge_padding_valid_mask",
            }
        )
    detail_dir = out_dir / "final_eval_details"
    save_detail_tables(val_records, detail_dir, "val")
    save_detail_tables(test_records, detail_dir, "test")
    write_json(
        {
            "val": val_metrics,
            "test": test_metrics,
            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "model_config": args._run_metadata["model_config"],
            "run_metadata": args._run_metadata,
        },
        out_dir / "final_metrics.json",
    )
    print(
        json.dumps(
            {"val": val_metrics, "test": test_metrics},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train/evaluate industrial liquid-level segmentation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser(
        "train_eval", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--root", default="data")
    p.add_argument(
        "--out_dir",
        default="outputs",
        help="base output directory; a timestamped subdirectory is auto-created under it",
    )
    p.add_argument(
        "--exp_name",
        default="",
        help="optional experiment name prefix (replaces auto-incremented expNNN)",
    )
    p.add_argument(
        "--model",
        default="phm_project_n2",
        choices=sorted(
            list(_MODEL_REGISTRY)
            + [
            "pidnet_s",
            "unet_resnet18_pretrained",
            "deeplabv3plus_resnet18_pretrained",
            "pspnet_resnet18_pretrained",
            "segformer_b0_pretrained",
            ]
        ),
    )
    p.add_argument("--condition_csv", default="")
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--video_weight_power", type=float, default=0.5)
    p.add_argument("--condition_weight_power", type=float, default=0.5)
    p.add_argument("--small_weight", type=float, default=2.0)
    p.add_argument(
        "--global_branch", default="avg", choices=["avg", "avgmax", "strip", "coord"]
    )
    p.add_argument(
        "--pidnet_pretrained",
        default="",
        help="Optional official PIDNet-S ImageNet-pretrained checkpoint.",
    )
    p.add_argument("--encoder_output_stride", type=int, default=8)
    p.add_argument("--aspp_out_ch", type=int, default=256)
    p.add_argument("--decoder_ch", type=int, default=256)
    p.add_argument("--low_ch", type=int, default=48)
    p.add_argument("--rates", type=int, nargs="+", default=[12, 24, 36])
    p.add_argument("--sk_reduction", type=int, default=16)
    p.add_argument("--phm_n", type=int, default=2)
    p.add_argument(
        "--run_script",
        default="",
        help="path to the invoking run script; its SHA256 is archived in artifacts",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="build model and print config/parameter counts without loading data or training",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train_eval)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
