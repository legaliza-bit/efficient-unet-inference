from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from loguru import logger

from src.data import get_voc_dataset
from src.utils import miou
from src.config import BATCH_SIZE, DEVICE, FINETUNE_EPOCHS, LR


criterion = nn.CrossEntropyLoss(ignore_index=255)


def finetune(model: nn.Module) -> None:
    full_train_ds = get_voc_dataset(root="data", image_set="train")

    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size

    generator = torch.Generator().manual_seed(42)

    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    model = model.to(DEVICE)

    for epoch in range(1, FINETUNE_EPOCHS + 1):

        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{FINETUNE_EPOCHS} [train]")

        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                preds = model(imgs)  # (B,C,H,W)
                loss = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):

                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    preds = model(imgs)

                    loss = criterion(preds, masks)
                    val_loss += loss.item()

                val_iou += miou(preds, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        logger.info(
            f"Epoch {epoch}/{FINETUNE_EPOCHS} — "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {avg_val_loss:.4f} | "
            f"val_mIoU: {avg_val_iou:.4f}"
        )

    ckpt_path = Path("./tmp/unet_finetuned.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Чекпоинт сохранён: {ckpt_path}")
