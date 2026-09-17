from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from framework.data import LiquidDataset
from framework.metrics import evaluate, save_detail_tables
from framework.utils import (
    choose_device,
    count_trainable_params,
    ensure_dir,
    generate_experiment_dir,
    write_json,
)
from models.factory import build_model_from_checkpoint


def evaluate_split(model, root: Path, split: str, args, device: torch.device):
    condition_csv = (
        Path(args.condition_csv)
        if args.condition_csv
        else root / "condition_image_level.csv"
    )
    ds = LiquidDataset(
        root,
        split,
        input_size=args.input_size,
        condition_csv=condition_csv,
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    metrics, records = evaluate(
        model, loader, device, threshold=args.threshold, amp=args.amp
    )
    return metrics, records


def cmd_evaluate(args) -> None:
    root = Path(args.root)
    device = choose_device(args.device)
    model, ckpt = build_model_from_checkpoint(Path(args.checkpoint), device)
    params = count_trainable_params(model)

    # 从 checkpoint 中提取模型名称和种子用于目录命名
    ckpt_model = (
        ckpt.get("model_name", "unknown") if isinstance(ckpt, dict) else "unknown"
    )
    ckpt_seed = int(ckpt.get("seed", 0)) if isinstance(ckpt, dict) else 0

    out_dir = generate_experiment_dir(
        args.out_dir, ckpt_model, ckpt_seed, args.exp_name
    )
    print(f"[INFO] evaluation directory: {out_dir}")

    splits = ["val", "test"] if args.split == "all" else [args.split]
    all_metrics = {}
    detail_dir = out_dir / "details"
    for split in splits:
        metrics, records = evaluate_split(model, root, split, args, device)
        metrics.update(
            {
                "split": split,
                "checkpoint": str(args.checkpoint),
                "params": params,
                "threshold": args.threshold,
                "preprocessing": "edge_padding_valid_mask",
            }
        )
        if isinstance(ckpt, dict):
            metrics.update(
                {
                    "checkpoint_epoch": ckpt.get("epoch"),
                    "checkpoint_score": ckpt.get("score"),
                    "checkpoint_model_name": ckpt.get("model_name"),
                    "checkpoint_global_branch": ckpt.get("global_branch"),
                    "checkpoint_phm_n": ckpt.get("phm_n"),
                }
            )
        save_detail_tables(records, detail_dir, split)
        all_metrics[split] = metrics

    write_json({"metrics": all_metrics, "args": vars(args)}, out_dir / "metrics.json")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2, default=str))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a PHMProject-n2 checkpoint on val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", default="data", help="dataset root containing imgs/ and masks/"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument(
        "--out_dir",
        default="evaluations",
        help="base output directory; a timestamped subdirectory is auto-created under it",
    )
    parser.add_argument(
        "--exp_name",
        default="",
        help="optional experiment name prefix (replaces auto-incremented expNNN)",
    )
    parser.add_argument("--split", default="test", choices=["val", "test", "all"])
    parser.add_argument("--condition_csv", default="")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(func=cmd_evaluate)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
