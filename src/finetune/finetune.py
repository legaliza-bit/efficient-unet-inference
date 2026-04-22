import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_carvana
from src.config import (
    BATCH_SIZE, FINETUNE_EPOCHS, QAT_EPOCHS, LR,
    DEVICE, NUM_CLASSES, CKPT_PATH, QAT_CKPT_PATH,
)
from src.finetune.losses import CombinedLoss
from src.metrics import mean_iou


def _make_loaders(num_workers=4):
    train_ds = get_carvana("train")
    val_ds = get_carvana("val")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


def _train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda" and scaler is not None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)
            loss = criterion(preds, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def _val_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_iou = 0, 0
    use_amp = device.type == "cuda"
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)
            loss = criterion(preds, masks)
        total_loss += loss.item()
        total_iou += mean_iou(preds, masks, num_classes)
    return total_loss / len(loader), total_iou / len(loader)


def finetune(model, num_classes=NUM_CLASSES, save_path=CKPT_PATH):
    train_loader, val_loader = _make_loaders()

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=FINETUNE_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1,
        div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler() if DEVICE.type == "cuda" else None

    model.to(DEVICE)

    for epoch in range(FINETUNE_EPOCHS):
        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, DEVICE
        )
        val_loss, val_iou = _val_epoch(model, val_loader, criterion, DEVICE, num_classes)
        print(
            f"Epoch {epoch:2d}: train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | mIoU={val_iou:.4f}"
        )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")


def finetune_qat(model, num_classes=NUM_CLASSES, save_path=QAT_CKPT_PATH):
    """Finetune with fake quantization nodes (QAT), convert to int8, save."""
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx  # noqa: PLC0415
    from torch.ao.quantization import get_default_qat_qconfig_mapping

    model = copy.deepcopy(model).cpu().train()

    dummy_h, dummy_w = 480, 640
    example_inputs = (torch.randn(1, 3, dummy_h, dummy_w),)
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

    train_loader, val_loader = _make_loaders(num_workers=2)

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR / 10, weight_decay=1e-4
    )

    cpu = torch.device("cpu")

    for epoch in range(QAT_EPOCHS):
        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, None, None, cpu
        )
        val_loss, val_iou = _val_epoch(
            model, val_loader, criterion, cpu, num_classes
        )
        print(
            f"QAT Epoch {epoch:2d}: train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | mIoU={val_iou:.4f}"
        )

    model = convert_fx(model.eval())

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, save_path)
        print(f"Saved QAT checkpoint to {save_path}")

    return model
