from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def generate_experiment_dir(
    base_dir: str | Path, model: str, seed: int, exp_name: str = ""
) -> Path:
    """生成带实验编号、模型、种子和时间标识的唯一输出目录。

    目录格式: {base_dir}/exp{NNN}_{model}_seed{seed}_{YYYYMMDD_HHMMSS}/
    若提供 exp_name 则用它替代 exp{NNN} 部分。
    实验编号 NNN 通过扫描 base_dir 下已有的 exp 开头的目录自动递增。
    """
    base = Path(base_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if exp_name:
        prefix = exp_name
    else:
        # 扫描已有目录，自动递增实验编号
        existing = []
        try:
            for entry in base.iterdir():
                if entry.is_dir():
                    m = re.match(r"^exp(\d+)_.+", entry.name)
                    if m:
                        existing.append(int(m.group(1)))
        except FileNotFoundError:
            pass
        next_num = max(existing) + 1 if existing else 1
        prefix = f"exp{next_num:03d}"

    dir_name = f"{prefix}_{model}_seed{seed}_{timestamp}"
    out_dir = base / dir_name
    ensure_dir(out_dir)
    return out_dir
