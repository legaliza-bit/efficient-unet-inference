import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.config import BATCH_SIZES, DEVICE, NUM_CLASSES, NUM_EVAL_SAMPLES
from src.data import get_fixed_eval_dataset
from src.models.baseline import get_pretrained_segmentation_model
from src.run_benchmark import run_benchmark
from src.utils import BenchmarkResult


def main() -> list[BenchmarkResult]:
    logger.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Memory: {mem:.1f} GB")

    logger.info("Loading VOC 2012 val dataset...")
    eval_ds = get_fixed_eval_dataset(root="data", num_samples=NUM_EVAL_SAMPLES)
    logger.info(f"Eval samples: {len(eval_ds)}")

    model = get_pretrained_segmentation_model(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: DeepLabV3 resnet50 | {n_params / 1e6:.1f}M params")

    if DEVICE.type == "cuda":
        model = model.half()

    results = []

    for bs in BATCH_SIZES:
        logger.info(f"\n--- batch_size={bs} ---")

        loader = DataLoader(
            eval_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        r = run_benchmark(
            model=model,
            dataloader=loader,
            device=DEVICE,
            pipeline_name=f"fp16_baseline_bs{bs}",
            num_classes=NUM_CLASSES,
        )
        results.append(r)

    print_summary(results)
    return results


def print_summary(results: list[BenchmarkResult]):
    header = (
        f"{'Pipeline':<28} {'BS':>3} {'Lat mean':>9} {'Lat p50':>8} "
        f"{'Lat p95':>8} {'Thpt':>8} {'mIoU':>7}"
    )
    sep = "-" * len(header)
    logger.info(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        m = f"{r.miou:.4f}" if r.miou is not None else "N/A"
        logger.info(
            f"{r.pipeline_name:<28} {r.batch_size:>3} "
            f"{r.latency_mean_ms:>8.2f}ms {r.latency_p50_ms:>7.2f}ms "
            f"{r.latency_p95_ms:>7.2f}ms {r.throughput_samples_per_sec:>7.1f} "
            f"{m:>7}"
        )
    logger.info(sep)


if __name__ == "__main__":
    main()
