from __future__ import annotations

import numpy as np
import torch

from .metrics import masked_bce_dice_loss


def train_one_epoch(model, loader, optimizer, device, amp: bool = False, scaler=None) -> float:
    model.train()
    losses = []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        valids = batch["valid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(imgs)
            loss = masked_bce_dice_loss(logits, masks, valids)
        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")
