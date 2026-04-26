"""Batch size sweep for fp32_baseline. Plots throughput, latency, and GPU memory vs batch size.

Usage:
    uv run python -m src.batch_size_study
    uv run python -m src.batch_size_study --batch-sizes 1 2 4 8 16 32 64
"""
import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.benchmark import run_benchmark
from src.config import CACHE_PATH, DEVICE, RESULTS_DIR
from src.data import CachedDataset, download_bench_cache
from src.model import load_model
from src.utils import set_seed

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="BS"
    )
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        download_bench_cache()
    dataset = CachedDataset(CACHE_PATH)
    model = load_model()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    batch_sizes, throughputs, lat_batch, lat_image, gpu_mems = [], [], [], [], []

    for bs in args.batch_sizes:
        if bs > len(dataset):
            break

        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda"),
            worker_init_fn=set_seed,
        )

        r = run_benchmark(model, dataloader, DEVICE, f"fp32_bs{bs}", "fp32")

        batch_sizes.append(bs)
        throughputs.append(r.throughput_samples_per_sec)
        lat_batch.append(r.latency_mean_ms)
        lat_image.append(r.latency_mean_ms / bs)
        gpu_mems.append(r.peak_gpu_memory_MB)

    _plot(batch_sizes, throughputs, lat_batch, lat_image, gpu_mems)


def _plot(batch_sizes, throughputs, lat_batch, lat_image, gpu_mems):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("FP32 Baseline — Batch Size Study", fontsize=14)

    kw = dict(marker="o", linewidth=2, markersize=6)

    axes[0, 0].plot(batch_sizes, throughputs, color="steelblue", **kw)
    axes[0, 0].set_title("Throughput")
    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].set_ylabel("img/s")

    axes[0, 1].plot(batch_sizes, gpu_mems, color="tomato", **kw)
    axes[0, 1].set_title("Peak GPU Memory")
    axes[0, 1].set_xlabel("Batch size")
    axes[0, 1].set_ylabel("MB")

    axes[1, 0].plot(batch_sizes, lat_batch, color="darkorange", **kw)
    axes[1, 0].set_title("Latency per Batch")
    axes[1, 0].set_xlabel("Batch size")
    axes[1, 0].set_ylabel("ms")

    axes[1, 1].plot(batch_sizes, lat_image, color="seagreen", **kw)
    axes[1, 1].set_title("Latency per Image")
    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylabel("ms/img")

    for ax in axes.flat:
        ax.set_xticks(batch_sizes)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "batch_size_study.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
