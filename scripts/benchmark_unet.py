import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import segmentation_models_pytorch as smp
import torch
from models import NUM_CLASSES, get_experiment
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
DATASET_NAME = "oxfordiiitpet"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 2
DEFAULT_NUM_SAMPLES = 64
DEFAULT_WARMUP_STEPS = 5
DEFAULT_SEED = 42


@dataclass
class PerfMetrics:
    num_images: int
    batch_size: int
    warmup_steps: int
    measured_steps: int
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    throughput_images_per_s: float
    max_memory_allocated_mb: float | None


@dataclass
class QualityMetrics:
    pixel_accuracy: float
    iou: float
    dice: float


@dataclass
class DebugMetrics:
    mean_pred_positive_ratio: float
    mean_target_positive_ratio: float
    logits_min: float
    logits_max: float
    logits_mean: float


class OxfordPetSegmentationDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int,
        download: bool,
        encoder_name: str,
        encoder_weights: str | None,
    ):
        self.preprocess_input = smp.encoders.get_preprocessing_fn(
            encoder_name=encoder_name,
            pretrained=encoder_weights,
        )
        self.image_resize = transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BILINEAR
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.NEAREST
                ),
                transforms.PILToTensor(),
            ]
        )
        self.dataset = datasets.OxfordIIITPet(
            root=str(root),
            split="test",
            target_types="segmentation",
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        image = self.image_resize(image)
        image = np.asarray(image).astype("float32")
        image = self.preprocess_input(image)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().contiguous()
        mask = self.mask_transform(mask).squeeze(0).long()
        mask = (mask != 2).long()
        return image, mask


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def build_dataset(
    num_samples: int,
    image_size: int,
    download: bool,
    encoder_name: str,
    encoder_weights: str | None,
) -> Dataset:
    dataset: Dataset = OxfordPetSegmentationDataset(
        root=DATA_DIR,
        image_size=image_size,
        download=download,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )
    return Subset(dataset, range(min(len(dataset), num_samples)))


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(sorted_values: List[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def update_confusion_matrix(
    confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor
) -> None:
    valid = (targets >= 0) & (targets < 2)
    preds = preds[valid]
    targets = targets[valid]
    indices = targets * 2 + preds
    confusion += torch.bincount(indices, minlength=4).reshape(2, 2)


def compute_quality_metrics(confusion: torch.Tensor) -> QualityMetrics:
    confusion = confusion.float()
    true_positive = confusion.diag()
    total = confusion.sum()
    row_sum = confusion.sum(dim=1)
    col_sum = confusion.sum(dim=0)

    positive_iou = true_positive[1] / (
        row_sum[1] + col_sum[1] - true_positive[1]
    ).clamp_min(1.0)
    positive_dice = 2.0 * true_positive[1] / (row_sum[1] + col_sum[1]).clamp_min(1.0)

    return QualityMetrics(
        pixel_accuracy=(true_positive.sum() / total.clamp_min(1.0)).item(),
        iou=positive_iou.item(),
        dice=positive_dice.item(),
    )


def run_benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    warmup_steps: int,
) -> Tuple[PerfMetrics, QualityMetrics, DebugMetrics]:
    latencies_ms: List[float] = []
    measured_images = 0
    measured_steps = 0
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    autocast_enabled = device.type == "cuda" and dtype in (
        torch.float16,
        torch.bfloat16,
    )
    pred_positive_sum = 0.0
    target_positive_sum = 0.0
    logits_min = math.inf
    logits_max = -math.inf
    logits_sum = 0.0
    logits_count = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for step, (images, masks) in enumerate(dataloader):
            start = time.perf_counter()
            images = images.to(device=device, dtype=dtype, non_blocking=True)
            masks = masks.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=autocast_enabled
            ):
                logits = model(images)
            synchronize(device)
            end = time.perf_counter()

            if logits.shape[1] != NUM_CLASSES:
                raise RuntimeError(
                    f"Expected {NUM_CLASSES} output channel, got {logits.shape[1]}."
                )
            logits_cpu = logits[:, 0].detach().float().cpu()
            predictions = (torch.sigmoid(logits[:, 0]) >= 0.5).long().cpu()
            targets_cpu = masks.cpu()
            update_confusion_matrix(confusion, predictions, targets_cpu)

            pred_positive_sum += predictions.float().mean().item()
            target_positive_sum += targets_cpu.float().mean().item()
            logits_min = min(logits_min, logits_cpu.min().item())
            logits_max = max(logits_max, logits_cpu.max().item())
            logits_sum += logits_cpu.sum().item()
            logits_count += logits_cpu.numel()

            if step >= warmup_steps:
                latencies_ms.append((end - start) * 1000.0)
                measured_images += images.size(0)
                measured_steps += 1

    if not latencies_ms:
        raise RuntimeError(
            "No measured iterations collected. Reduce --warmup-steps or increase --num-samples."
        )

    sorted_latencies = sorted(latencies_ms)
    total_seconds = sum(latencies_ms) / 1000.0
    max_memory_allocated_mb = None
    if device.type == "cuda":
        max_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    perf_metrics = PerfMetrics(
        num_images=measured_images,
        batch_size=dataloader.batch_size or 0,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        mean_latency_ms=sum(latencies_ms) / len(latencies_ms),
        median_latency_ms=percentile(sorted_latencies, 0.5),
        p90_latency_ms=percentile(sorted_latencies, 0.9),
        throughput_images_per_s=(
            measured_images / total_seconds if total_seconds > 0 else math.nan
        ),
        max_memory_allocated_mb=max_memory_allocated_mb,
    )
    quality_metrics = compute_quality_metrics(confusion)
    debug_metrics = DebugMetrics(
        mean_pred_positive_ratio=pred_positive_sum / len(dataloader),
        mean_target_positive_ratio=target_positive_sum / len(dataloader),
        logits_min=logits_min,
        logits_max=logits_max,
        logits_mean=logits_sum / logits_count,
    )
    return perf_metrics, quality_metrics, debug_metrics


@click.command()
@click.option(
    "--exp-name",
    type=click.Choice(["baseline_fp32", "baseline_fp16", "compile"], case_sensitive=True),
    default="baseline_fp16",
    show_default=True,
)
@click.option("--download", is_flag=True, default=False)
@click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True)
@click.option("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, show_default=True)
@click.option("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, show_default=True)
@click.option("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, show_default=True)
@click.option(
    "--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, show_default=True
)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
def main(
    exp_name: str,
    download: bool,
    batch_size: int,
    num_workers: int,
    image_size: int,
    num_samples: int,
    warmup_steps: int,
    device: str,
    seed: int,
) -> None:
    set_seed(seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device_obj = resolve_device(device)
    experiment = get_experiment(exp_name)
    dataset = build_dataset(
        num_samples=num_samples,
        image_size=image_size,
        download=download,
        encoder_name=experiment.encoder_name,
        encoder_weights=experiment.encoder_weights,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = experiment.build_model(device_obj)

    perf_metrics, quality_metrics, debug_metrics = run_benchmark(
        model=model,
        dataloader=dataloader,
        device=device_obj,
        dtype=experiment.dtype,
        warmup_steps=warmup_steps,
    )

    result: Dict[str, object] = {
        "exp_name": experiment.name,
        "dataset": DATASET_NAME,
        "model": {
            "architecture": experiment.architecture,
            "encoder_name": experiment.encoder_name,
            "encoder_weights": experiment.encoder_weights,
            "num_classes": NUM_CLASSES,
        },
        "runtime": {
            "device": str(device_obj),
            "dtype": str(experiment.dtype),
            "image_size": image_size,
            "num_samples": len(dataset),
        },
        "perf_metrics": asdict(perf_metrics),
        "quality_metrics": asdict(quality_metrics),
        "debug_metrics": asdict(debug_metrics),
    }

    output_path = ARTIFACTS_DIR / f"{experiment.name}.json"
    output_path.write_text(json.dumps(result, indent=2))
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
