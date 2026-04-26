import copy
from typing import Callable

import torch
from loguru import logger
from torchao.quantization import (
    Float8WeightOnlyConfig,
    Int8StaticActivationInt8WeightConfig,
    quantize_,
)

from src.config import CKPT_PATH, IMG_SCALE


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


def apply_pruning(model, sparsity: float = 0.5):
    """Global unstructured magnitude pruning across all Conv2d weights."""
    from torch.nn.utils import prune
    model = copy.deepcopy(model).cuda().eval()
    params = [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
    for m, _ in params:
        prune.remove(m, "weight")
    return model


def apply_sparse_2_4(model):
    """Apply 2:4 structured sparsity mask to all Conv2d weights (no runtime speedup — Conv2d not supported by cuSPARSELt)."""
    model = copy.deepcopy(model).cuda().eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            with torch.no_grad():
                w = m.weight.data
                shape = w.shape
                w_flat = w.view(-1, 4)
                mask = torch.zeros_like(w_flat)
                mask.scatter_(1, w_flat.abs().topk(2, dim=1).indices, 1)
                m.weight.copy_((w_flat * mask).view(shape))
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
    """FP8 weight-only quantization via torchao (requires SM 8.9+, e.g. RTX 4090)."""
    model = copy.deepcopy(model).cuda().eval()
    quantize_(model, Float8WeightOnlyConfig())
    return model
