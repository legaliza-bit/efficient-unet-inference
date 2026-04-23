import copy
import torch
from torchao.quantization import quantize_, Float8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig

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


def apply_compiled(model):
    """torch.compile with max-autotune (no CUDA graphs to avoid private pool OOM)."""
    return torch.compile(model, mode="max-autotune-no-cudagraphs")


def apply_int8(model):
    """INT8 dynamic activation + weight quantization via torchao (GPU)."""
    model = copy.deepcopy(model).cuda().eval()
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    return model


def apply_fp8(model):
    """FP8 weight-only quantization via torchao (requires SM 8.9+, e.g. RTX 4090)."""
    model = copy.deepcopy(model).cuda().eval()
    quantize_(model, Float8WeightOnlyConfig())
    return model
