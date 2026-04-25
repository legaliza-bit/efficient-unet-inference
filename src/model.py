import copy
import torch
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8StaticActivationInt8WeightConfig

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


def apply_int8(model, calib_dataloader=None):
    """INT8 static activation + weight quantization via torchao (GPU).

    If *calib_dataloader* is provided, a few batches are forwarded to
    collect activation statistics for static quantization.
    """
    model = copy.deepcopy(model).cuda().eval()
    quantize_(model, Int8StaticActivationInt8WeightConfig())
    # Calibration: run representative data to collect activation statistics
    if calib_dataloader is not None:
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_dataloader):
                images = images.cuda()
                model(images)
                if i >= 2:  # A few batches is enough for calibration
                    break
    return model


def apply_fp8(model):
    """FP8 dynamic activation + weight quantization via torchao (requires SM 8.9+, e.g. RTX 4090)."""
    model = copy.deepcopy(model).cuda().eval()
    quantize_(model, Float8DynamicActivationFloat8WeightConfig())
    return model
