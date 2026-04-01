import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data import get_voc_dataset
from src.config import BATCH_SIZE, FINETUNE_EPOCHS, LR, DEVICE


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        probs = torch.softmax(logits, dim=1)

        mask = targets != self.ignore_index
        targets_clamped = targets.clone()
        targets_clamped[~mask] = 0

        targets_onehot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()

        probs = probs * mask.unsqueeze(1)
        targets_onehot = targets_onehot * mask.unsqueeze(1)

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits, targets):
        return self.w_ce * self.ce(logits, targets) + self.w_dice * self.dice(logits, targets)


@torch.no_grad()
def mean_iou(preds, targets, num_classes, ignore_index=255):
    preds = torch.argmax(preds, dim=1)

    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_mask = preds == cls
        true_mask = targets == cls

        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()

        if union == 0:
            continue

        ious.append((intersection / union).item())

    return sum(ious) / max(len(ious), 1)


def finetune(model, num_classes=21):

    full_train_ds = get_voc_dataset(image_set="train")
    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size

    generator = torch.Generator().manual_seed(42)

    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Loaded train: {len(train_ds)} | val: {len(val_ds)} samples")

    criterion = CombinedLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=FINETUNE_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

    scaler = torch.amp.GradScaler()

    model.to(DEVICE)

    for name, p in model.named_parameters():
        if "enc" in name:
            p.requires_grad = False

    print("Stage 1: encoder frozen")

    for epoch in range(FINETUNE_EPOCHS):
        model.train()

        train_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                preds = model(imgs)
                loss = criterion(preds, masks)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        model.eval()

        val_loss = 0
        val_iou = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):

                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                with torch.amp.autocast(device_type="cuda"):
                    preds = model(imgs)
                    loss = criterion(preds, masks)

                val_loss += loss.item()
                val_iou += mean_iou(preds, masks, num_classes)

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={train_loss/len(train_loader):.4f} | "
            f"val_loss={val_loss/len(val_loader):.4f} | "
            f"mIoU={val_iou/len(val_loader):.4f}\n"
        )

        if epoch == FINETUNE_EPOCHS // 2:
            for p in model.parameters():
                p.requires_grad = True
            print("Stage 2: encoder unfrozen")
