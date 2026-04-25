import copy
import torch
from loguru import logger
from torchao.quantization import quantize_, Float8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig

from src.config import IMG_SCALE, CKPT_PATH
from typing import Callable


def load_model(pretrained=True):
    """Load milesial/Pytorch-UNet pretrained on Carvana."""
    if CKPT_PATH.exists():
        model = torch.hub.load(
            "milesial/Pytorch-UNet",
            "unet_carvana",
            pretrained=False,
            scale=IMG_SCALE,
        )
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        logger.info(f"Loaded checkpoint: {CKPT_PATH}")
    else:
        model = torch.hub.load(
            "milesial/Pytorch-UNet",
            "unet_carvana",
            pretrained=pretrained,
            scale=IMG_SCALE,
        )
        print("Loaded pretrained U-Net")
    return model


def warmup_model(
    model: Callable,
    dummy_input: torch.Tensor,
    n_iters: int = 10,
    device: torch.device | None = None,
    precision: str = "fp32",
) -> None:
    use_amp = precision != "fp32" and device is not None and device.type == "cuda"
    with torch.no_grad():
        for _ in range(n_iters):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    if device and device.type == "cuda":
        torch.cuda.synchronize(device)


def forward(model, x: torch.Tensor, precision: str) -> torch.Tensor:
    if precision != "fp32" and x.device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return model(x)
    return model(x)


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
