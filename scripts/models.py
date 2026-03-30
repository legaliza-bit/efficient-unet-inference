from dataclasses import dataclass
from typing import Callable

import segmentation_models_pytorch as smp
import torch
from torch import nn

PRETRAINED_ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 1
PRETRAINED_CHECKPOINT = "SeppSepp/oxford-pet-segmentation"


@dataclass(frozen=True)
class Experiment:
    name: str
    dtype: torch.dtype
    build_model: Callable[[torch.device], nn.Module]
    architecture: str
    encoder_name: str
    encoder_weights: str | None


def build_pretrained_oxford_pet_unet_fp16(device: torch.device) -> nn.Module:
    model = smp.from_pretrained(PRETRAINED_CHECKPOINT)
    model.eval()
    model.to(device=device)
    model.to(dtype=torch.float16)
    return model


def build_compiled_pretrained_oxford_pet_unet_fp16(device: torch.device) -> nn.Module:
    model = build_pretrained_oxford_pet_unet_fp16(device)
    return torch.compile(model, backend="inductor", mode="max-autotune")


def get_experiment(exp_name: str) -> Experiment:
    experiments = {
        "baseline_fp16": Experiment(
            name="baseline_fp16",
            dtype=torch.float16,
            build_model=build_pretrained_oxford_pet_unet_fp16,
            architecture="smp.Unet",
            encoder_name=PRETRAINED_ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
        ),
        "compile": Experiment(
            name="compile",
            dtype=torch.float16,
            build_model=build_compiled_pretrained_oxford_pet_unet_fp16,
            architecture="smp.Unet",
            encoder_name=PRETRAINED_ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
        ),
    }
    if exp_name not in experiments:
        available = ", ".join(sorted(experiments))
        raise ValueError(f"Unknown exp_name={exp_name}. Available: {available}")
    return experiments[exp_name]
