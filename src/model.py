import copy
import torch
import torch.nn as nn

from src.config import IMG_SCALE


_CHECKPOINTS = {
    0.5: "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    1.0: "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth",
}


def build_model(scale=IMG_SCALE):
    """Load milesial/Pytorch-UNet pretrained on Carvana, always to CPU."""
    model = torch.hub.load(
        "milesial/Pytorch-UNet",
        "unet_carvana",
        pretrained=False,
        scale=scale,
    )
    if scale not in _CHECKPOINTS:
        raise ValueError(f"No pretrained checkpoint for scale={scale}. Use 0.5 or 1.0.")
    state_dict = torch.hub.load_state_dict_from_url(
        _CHECKPOINTS[scale], map_location="cpu", progress=True
    )
    state_dict.pop("mask_values", None)
    model.load_state_dict(state_dict)
    return model


def apply_ptq(model):
    model = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
