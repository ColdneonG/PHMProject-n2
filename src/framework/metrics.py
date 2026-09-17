from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ensure_dir


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def masked_bce_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    valid = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    dims = (1, 2, 3)
    inter = (probs * targets * valid).sum(dims)
    denom = (probs * valid).sum(dims) + (targets * valid).sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def binary_metrics(
    pred: np.ndarray, target: np.ndarray, eps: float = 1e-6
) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    inter = tp
    union = np.logical_or(pred, target).sum()
    return {
        "dice": float((2 * inter + eps) / (pred.sum() + target.sum() + eps)),
        "iou": float((inter + eps) / (union + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "pred_area": float(pred.mean()),
        "target_area": float(target.mean()),
    }


def boundary_map(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = mask.astype(np.uint8)
    try:
        import cv2

        kernel = np.ones((3, 3), np.uint8)
        return (
            cv2.dilate(mask, kernel, iterations=radius)
            - cv2.erode(mask, kernel, iterations=radius)
        ) > 0
    except Exception:
        b = np.zeros_like(mask, dtype=bool)
        b[1:, :] |= mask[1:, :] != mask[:-1, :]
        b[:-1, :] |= mask[:-1, :] != mask[1:, :]
        b[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        return b


def dilate_bool(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    mask = mask.astype(np.uint8)
    try:
        import cv2

        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1).astype(bool)
    except Exception:
        out = mask.astype(bool)
        for _ in range(radius):
            pad = np.pad(out, ((1, 1), (1, 1)), mode="constant")
            out = (
                pad[1:-1, 1:-1]
                | pad[:-2, 1:-1]
                | pad[2:, 1:-1]
                | pad[1:-1, :-2]
                | pad[1:-1, 2:]
            )
        return out


def boundary_f1(
    pred: np.ndarray, target: np.ndarray, tol: int = 2, eps: float = 1e-6
) -> float:
    pb = boundary_map(pred)
    tb = boundary_map(target)
    if pb.sum() == 0 and tb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or tb.sum() == 0:
        return 0.0
    precision = (pb & dilate_bool(tb, tol)).sum() / pb.sum()
    recall = (tb & dilate_bool(pb, tol)).sum() / tb.sum()
    return float(2 * precision * recall / (precision + recall + eps))


def liquid_level(mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.nan
    h, _ = mask.shape
    cols = np.where(mask.any(axis=0))[0]
    top_ys = []
    for x in cols:
        ys = np.where(mask[:, x])[0]
        if len(ys):
            top_ys.append(ys.min())
    if not top_ys:
        return np.nan
    return float(np.mean(top_ys) / max(h - 1, 1))


def mean_or_nan(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    out = {
        "num_samples": len(records),
        "overall_dice": mean_or_nan([r["dice"] for r in records]),
        "overall_iou": mean_or_nan([r["iou"] for r in records]),
        "overall_recall": mean_or_nan([r["recall"] for r in records]),
        "overall_precision": mean_or_nan([r["precision"] for r in records]),
        "small_dice": mean_or_nan(
            [
                r["dice"]
                for r in records
                if r["condition"] == "small" or r["group"] == "small"
            ]
        ),
        "small_iou": mean_or_nan(
            [
                r["iou"]
                for r in records
                if r["condition"] == "small" or r["group"] == "small"
            ]
        ),
        "small_recall": mean_or_nan(
            [
                r["recall"]
                for r in records
                if r["condition"] == "small" or r["group"] == "small"
            ]
        ),
        "boundary_f1": mean_or_nan([r["boundary_f1"] for r in records]),
        "small_boundary_f1": mean_or_nan(
            [
                r["boundary_f1"]
                for r in records
                if r["condition"] == "small" or r["group"] == "small"
            ]
        ),
        "level_mae": mean_or_nan([r["level_mae"] for r in records]),
        "small_level_mae": mean_or_nan(
            [
                r["level_mae"]
                for r in records
                if r["condition"] == "small" or r["group"] == "small"
            ]
        ),
        "nonempty_miss_rate": mean_or_nan(
            [float(r["pred_area"] < 0.01) for r in records]
        ),
        "empty_count": 0,
        "empty_fpr": float("nan"),
        "mean_pred_area": mean_or_nan([r["pred_area"] for r in records]),
        "mean_target_area": mean_or_nan([r["target_area"] for r in records]),
    }
    for cond in ["clear", "blur", "small", "unknown"]:
        sub = [r for r in records if r["condition"] == cond]
        out[f"condition_{cond}_dice"] = mean_or_nan([r["dice"] for r in sub])
        out[f"condition_{cond}_iou"] = mean_or_nan([r["iou"] for r in sub])
        out[f"condition_{cond}_recall"] = mean_or_nan([r["recall"] for r in sub])
        out[f"condition_{cond}_boundary_f1"] = mean_or_nan(
            [r["boundary_f1"] for r in sub]
        )
        out[f"condition_{cond}_level_mae"] = mean_or_nan([r["level_mae"] for r in sub])
    out["clear_dice"] = out["condition_clear_dice"]
    out["blur_dice"] = out["condition_blur_dice"]
    by_video = defaultdict(list)
    for record in records:
        by_video[record["video"]].append(record)
    video_dices = [mean_or_nan([r["dice"] for r in rs]) for rs in by_video.values()]
    out["worst_video_overall_dice"] = (
        float(np.nanmin(video_dices)) if video_dices else float("nan")
    )
    out["worst_video_nonempty_dice"] = out["worst_video_overall_dice"]
    level_mae = out["level_mae"] if not np.isnan(out["level_mae"]) else 1.0
    out["selection_score"] = float(
        0.50 * (out["overall_dice"] if not np.isnan(out["overall_dice"]) else 0.0)
        + 0.20 * (out["small_dice"] if not np.isnan(out["small_dice"]) else 0.0)
        + 0.15
        * (
            out["condition_blur_dice"]
            if not np.isnan(out["condition_blur_dice"])
            else 0.0
        )
        + 0.10 * (out["boundary_f1"] if not np.isnan(out["boundary_f1"]) else 0.0)
        - 0.05 * level_mae
    )
    return out


def save_detail_tables(
    records: list[dict[str, Any]], out_dir: Path, split: str
) -> None:
    ensure_dir(out_dir)
    write_csv_rows(out_dir / f"{split}_sample_metrics.csv", records)
    if not records:
        return
    conds = sorted({r["condition"] for r in records})
    cond_rows = [
        {
            "condition": cond,
            **summarize_records([r for r in records if r["condition"] == cond]),
        }
        for cond in conds
    ]
    write_csv_rows(out_dir / f"{split}_by_condition.csv", cond_rows)
    videos = sorted({r["video"] for r in records})
    video_rows = [
        {
            "video": video,
            **summarize_records([r for r in records if r["video"] == video]),
        }
        for video in videos
    ]
    write_csv_rows(out_dir / f"{split}_by_video.csv", video_rows)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
    amp: bool = False,
):
    model.eval()
    records = []
    n_imgs_fps = 0
    t_fps = 0.0
    for batch_idx, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        valids = batch["valid"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast(
            device_type=device.type, enabled=amp and device.type == "cuda"
        ):
            logits = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if batch_idx < 20:
            t_fps += time.time() - t0
            n_imgs_fps += imgs.size(0)
        preds = (torch.sigmoid(logits) >= threshold).float()
        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()
        valids_np = valids.cpu().numpy()
        for i in range(imgs.size(0)):
            valid = valids_np[i, 0] > 0.5
            pred = (preds_np[i, 0] > 0.5) & valid
            target = (masks_np[i, 0] > 0.5) & valid
            ys, xs = np.where(valid)
            if len(ys) > 0:
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                pred_c = pred[y0:y1, x0:x1]
                target_c = target[y0:y1, x0:x1]
            else:
                pred_c, target_c = pred, target
            bm = binary_metrics(pred_c, target_c)
            lp = liquid_level(pred_c)
            lt = liquid_level(target_c)
            records.append(
                {
                    "path": batch["path"][i],
                    "video": batch["video"][i],
                    "condition": batch["condition"][i],
                    "group": batch["group"][i],
                    "dice": bm["dice"],
                    "iou": bm["iou"],
                    "recall": bm["recall"],
                    "precision": bm["precision"],
                    "pred_area": bm["pred_area"],
                    "target_area": bm["target_area"],
                    "boundary_f1": boundary_f1(pred_c, target_c, tol=2),
                    "level_mae": float(abs(lp - lt))
                    if not (np.isnan(lp) or np.isnan(lt))
                    else float("nan"),
                }
            )
    metrics = summarize_records(records)
    metrics["fps"] = (
        float(n_imgs_fps / max(t_fps, 1e-8)) if n_imgs_fps > 0 else float("nan")
    )
    return metrics, records
