import torch


@torch.no_grad()
def mean_iou(preds, targets, num_classes, ignore_index=255):
    """Batch-level mIoU from raw logits. Used during training validation."""
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        true_mask = (targets == cls) & (targets != ignore_index)
        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return sum(ious) / max(len(ious), 1)


def update_conf_matrix(conf_matrix, preds, targets, num_classes):
    """Accumulate predictions into a confusion matrix in-place."""
    mask = (targets >= 0) & (targets < num_classes) & (targets != 255)
    conf_matrix += torch.bincount(
        (num_classes * targets[mask] + preds[mask]).view(-1),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def compute_miou_dice(conf_matrix):
    """Compute mIoU and mean Dice from an accumulated confusion matrix."""
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    miou = (intersection / (union + 1e-6)).mean().item()

    denom = conf_matrix.sum(1) + conf_matrix.sum(0) + 1e-6
    dice = ((2 * intersection) / denom).mean().item()

    return miou, dice
