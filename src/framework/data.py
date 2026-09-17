from __future__ import annotations

import random
import re
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_video_id(path_or_name: str) -> str:
    text = str(path_or_name)
    match = re.search(r"(video\d+_\d+-\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(video\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown_video"


def parse_physical_video(video_id: str) -> str:
    match = re.match(r"(video\d+)", video_id, flags=re.IGNORECASE)
    return match.group(1) if match else video_id


def condition_from_fg_ratio(mask_arr: np.ndarray, small_thr: float = 0.10) -> tuple[str, float]:
    fg = (mask_arr > 127).astype(np.uint8)
    ratio = float(fg.mean())
    if ratio <= 0:
        return "empty", ratio
    if ratio <= small_thr:
        return "small", ratio
    return "normal", ratio


def load_condition_map(csv_path: Optional[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if csv_path is None or not csv_path.exists():
        return mapping

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with csv_path.open("r", encoding="gbk", newline="") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        return mapping
    columns = list(rows[0].keys())

    valid_conditions = {"clear", "blur", "small", "empty"}
    cond_col = None
    for col in columns:
        name = str(col).strip().lower()
        if name in {"condition", "scene", "label", "class", "category", "场景", "类别", "标签"}:
            cond_col = col
            break
    if cond_col is None:
        best_col, best_hits = None, -1
        for col in columns:
            hits = sum(str(row.get(col, "")).strip().lower() in valid_conditions for row in rows)
            if hits > best_hits:
                best_col, best_hits = col, hits
        if best_hits > 0:
            cond_col = best_col
    if cond_col is None:
        print(f"[WARN] {csv_path} has no recognizable condition column: {columns}")
        return mapping

    split_col = None
    for col in columns:
        if str(col).strip().lower() in {"split", "subset", "set", "划分", "数据集"}:
            split_col = col
            break

    for row in rows:
        cond = str(row.get(cond_col, "")).strip().lower()
        if cond not in valid_conditions:
            continue
        split_val = str(row.get(split_col, "")).strip() if split_col is not None else ""
        keys = set()
        for col in columns:
            if col == cond_col:
                continue
            val = str(row.get(col, "")).strip()
            if not val or val.lower() in {"nan", "none", "null"}:
                continue
            norm = val.replace("\\", "/")
            keys.add(norm)
            p = Path(norm)
            if p.name:
                keys.add(p.name)
            if p.stem:
                keys.add(p.stem)
            parts = [x for x in norm.split("/") if x]
            for k in range(1, min(5, len(parts)) + 1):
                keys.add("/".join(parts[-k:]))
            if split_val and p.name:
                keys.add(f"{split_val}/{p.name}")
                keys.add(f"imgs/{split_val}/{p.name}")
            if split_val and p.stem:
                keys.add(f"{split_val}/{p.stem}")
                keys.add(f"imgs/{split_val}/{p.stem}")
        for key in keys:
            mapping[key] = cond
    print(f"[INFO] loaded {len(mapping)} condition keys from {csv_path}")
    return mapping


def lookup_condition(condition_map: dict[str, str], img_path: Path, root: Path, split: str, fallback: str) -> str:
    if not condition_map:
        return fallback
    candidates = []
    for base in [root, root / "imgs" / split]:
        try:
            candidates.append(str(img_path.relative_to(base)).replace("\\", "/"))
        except ValueError:
            pass
    candidates.extend([img_path.name, img_path.stem, str(img_path).replace("\\", "/")])
    return next((condition_map[k] for k in candidates if k in condition_map), fallback)


def collect_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    img_dir = root / "imgs" / split
    mask_dir = root / "masks" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"image directory not found: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask directory not found: {mask_dir}")

    img_files = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    mask_files = [p for p in mask_dir.rglob("*") if p.is_file() and p.suffix.lower() in MASK_EXTS]
    mask_by_key = {}
    for mask in mask_files:
        rel = mask.relative_to(mask_dir)
        mask_by_key[str(rel.with_suffix("")).replace("\\", "/")] = mask
        mask_by_key[mask.stem] = mask

    pairs = []
    for img in sorted(img_files):
        rel = img.relative_to(img_dir)
        for key in [str(rel.with_suffix("")).replace("\\", "/"), img.stem]:
            if key in mask_by_key:
                pairs.append((img, mask_by_key[key]))
                break
    if not pairs:
        raise RuntimeError(f"no image-mask pairs found for split={split}")
    return pairs


class ResizePadEdgeValid:
    """Aspect-ratio resize with edge-padded image, zero-padded mask and valid mask."""

    def __init__(self, size: int = 256):
        self.size = int(size)

    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_r = img.resize((nw, nh), Image.BILINEAR)
        mask_r = mask.resize((nw, nh), Image.NEAREST)
        left = (self.size - nw) // 2
        top = (self.size - nh) // 2
        right = self.size - nw - left
        bottom = self.size - nh - top

        img_np = np.asarray(img_r).astype(np.uint8)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_pad = np.pad(img_np, ((top, bottom), (left, right), (0, 0)), mode="edge")

        mask_np = np.asarray(mask_r).astype(np.uint8)
        mask_pad = np.pad(mask_np, ((top, bottom), (left, right)), mode="constant", constant_values=0)
        valid = np.zeros((self.size, self.size), dtype=np.uint8)
        valid[top : top + nh, left : left + nw] = 255
        return Image.fromarray(img_pad), Image.fromarray(mask_pad), Image.fromarray(valid)


class LiquidDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        input_size: int = 256,
        condition_csv: Optional[Path] = None,
        train: bool = False,
        small_thr: float = 0.10,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.root = Path(root)
        self.split = split
        self.pairs = collect_pairs(self.root, split)
        self.train = train
        self.resize_pad = ResizePadEdgeValid(input_size)
        self.mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
        if condition_csv is None:
            default_csv = self.root / "condition_image_level.csv"
            condition_csv = default_csv if default_csv.exists() else None
        self.condition_map = load_condition_map(condition_csv)
        self.meta = []
        for img, mask in self.pairs:
            mask_arr = np.array(Image.open(mask).convert("L"))
            auto_group, fg_ratio = condition_from_fg_ratio(mask_arr, small_thr=small_thr)
            fallback = "small" if auto_group == "small" else "clear"
            cond = lookup_condition(self.condition_map, img, self.root, split, fallback=fallback)
            if cond == "empty":
                cond = fallback
            video = parse_video_id(str(img))
            self.meta.append({
                "image": img,
                "mask": mask,
                "condition": cond,
                "group": "small" if auto_group == "small" else "normal",
                "fg_ratio": fg_ratio,
                "video": video,
                "physical_video": parse_physical_video(video),
            })

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.25:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.25:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
        return img, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.meta[idx]
        img = Image.open(meta["image"]).convert("RGB")
        mask = Image.open(meta["mask"]).convert("L")
        if self.train:
            img, mask = self._augment(img, mask)
        img, mask, valid = self.resize_pad(img, mask)
        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        mask_np = (np.asarray(mask) > 127).astype(np.float32)
        valid_np = (np.asarray(valid) > 127).astype(np.float32)
        return {
            "image": torch.from_numpy(img_np.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask_np[None]).float(),
            "valid": torch.from_numpy(valid_np[None]).float(),
            "condition": meta["condition"],
            "group": meta["group"],
            "video": meta["video"],
            "fg_ratio": float(meta["fg_ratio"]),
            "path": str(meta["image"]),
        }


def make_sampler(
    dataset: LiquidDataset,
    video_weight_power: float = 0.5,
    condition_weight_power: float = 0.5,
    small_weight: float = 2.0,
) -> WeightedRandomSampler:
    videos = [m["physical_video"] for m in dataset.meta]
    conds = [m["condition"] for m in dataset.meta]
    groups = [m["group"] for m in dataset.meta]
    vc, cc = Counter(videos), Counter(conds)
    weights = []
    for video, cond, group in zip(videos, conds, groups):
        w = 1.0
        if video_weight_power > 0:
            w *= (1.0 / max(vc[video], 1)) ** video_weight_power
        if condition_weight_power > 0:
            w *= (1.0 / max(cc[cond], 1)) ** condition_weight_power
        if group == "small":
            w *= small_weight
        weights.append(w)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
