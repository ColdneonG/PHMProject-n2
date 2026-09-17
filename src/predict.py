from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from framework.data import IMG_EXTS, ResizePadEdgeValid
from framework.utils import choose_device, ensure_dir, generate_experiment_dir
from models.factory import build_model_from_checkpoint


def preprocess_image(path: Path, input_size: int):
    img = Image.open(path).convert("RGB")
    dummy_mask = Image.fromarray(np.zeros((img.height, img.width), dtype=np.uint8))
    img_pad, _, valid = ResizePadEdgeValid(input_size)(img, dummy_mask)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    arr = np.asarray(img_pad).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_pad, (np.asarray(valid) > 127), tensor


def save_overlay(img_pad: Image.Image, pred: np.ndarray, out_path: Path) -> None:
    rgb = np.asarray(img_pad).copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    overlay = np.where(pred[..., None], (0.65 * rgb + 0.35 * red).astype(np.uint8), rgb)
    Image.fromarray(overlay).save(out_path)


@torch.no_grad()
def cmd_predict(args):
    device = choose_device(args.device)
    model, ckpt = build_model_from_checkpoint(Path(args.checkpoint), device)

    # 从 checkpoint 中提取模型名称和种子用于目录命名
    ckpt_model = (
        ckpt.get("model_name", "unknown") if isinstance(ckpt, dict) else "unknown"
    )
    ckpt_seed = int(ckpt.get("seed", 0)) if isinstance(ckpt, dict) else 0

    out_dir = generate_experiment_dir(
        args.out_dir, ckpt_model, ckpt_seed, args.exp_name
    )
    print(f"[INFO] prediction directory: {out_dir}")
    input_path = Path(args.input)
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)

    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            [
                p
                for p in input_path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ]
        )
    else:
        raise FileNotFoundError(f"input path not found: {input_path}")
    if not images:
        raise RuntimeError(f"no image files found under: {input_path}")
    rows = []
    for img_path in images:
        img_pad, valid, tensor = preprocess_image(img_path, args.input_size)
        tensor = tensor.to(device)
        with torch.amp.autocast(
            device_type=device.type, enabled=args.amp and device.type == "cuda"
        ):
            logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = ((prob >= args.threshold) & valid).astype(np.uint8)
        rel_name = (
            img_path.stem
            if input_path.is_file()
            else "__".join(img_path.relative_to(input_path).with_suffix("").parts)
        )
        mask_path = mask_dir / f"{rel_name}.png"
        overlay_path = overlay_dir / f"{rel_name}.png"
        Image.fromarray(pred * 255).save(mask_path)
        save_overlay(img_pad, pred.astype(bool), overlay_path)
        rows.append(
            {
                "image": str(img_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "pred_area_valid": float(pred[valid].mean()) if valid.any() else 0.0,
            }
        )
    with (out_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "mask", "overlay", "pred_area_valid"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] predicted {len(rows)} images with checkpoint={args.checkpoint}")
    if isinstance(ckpt, dict):
        print(
            {
                k: ckpt.get(k)
                for k in ["model_name", "epoch", "score", "global_branch", "phm_n"]
            }
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run PHMProject-n2 checkpoint inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="data/imgs/test", help="image file or directory"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument(
        "--out_dir",
        default="predictions",
        help="base output directory; a timestamped subdirectory is auto-created under it",
    )
    parser.add_argument(
        "--exp_name",
        default="",
        help="optional experiment name prefix (replaces auto-incremented expNNN)",
    )
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(func=cmd_predict)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
