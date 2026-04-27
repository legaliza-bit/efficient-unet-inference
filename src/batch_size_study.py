"""Batch size sweep for fp32_baseline. Plots throughput, latency, GPU memory, and Pareto front.

Usage:
    uv run python -m src.batch_size_study
    uv run python -m src.batch_size_study --batch-sizes 1 2 4 8 16 32 64
"""
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.benchmark import run_benchmark
from src.config import CACHE_PATH, DEVICE, RESULTS_DIR
from src.data import CachedDataset, download_bench_cache
from src.model import load_model
from src.utils import set_seed

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="BS"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE)

    if not CACHE_PATH.exists():
        download_bench_cache()
    dataset = CachedDataset(CACHE_PATH)
    model = load_model()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    batch_sizes, throughputs, lat_batch, gpu_mems = [], [], [], []

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
        gpu_mems.append(r.peak_gpu_memory_MB)

    _plot(batch_sizes, throughputs, lat_batch, gpu_mems)


def _plot(batch_sizes, throughputs, lat_batch, gpu_mems):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("FP32 Baseline — Batch Size Study", fontsize=14, fontweight="bold")

    kw = dict(marker="o", linewidth=2, markersize=7)
    color = sns.color_palette("muted")

    # Throughput
    axes[0, 0].plot(batch_sizes, throughputs, color=color[0], **kw)
    axes[0, 0].set_title("Throughput")
    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].set_ylabel("img/s")

    # Peak GPU memory
    axes[0, 1].plot(batch_sizes, gpu_mems, color=color[1], **kw)
    axes[0, 1].set_title("Peak GPU Memory")
    axes[0, 1].set_xlabel("Batch size")
    axes[0, 1].set_ylabel("MB")

    # Latency per batch
    axes[1, 0].plot(batch_sizes, lat_batch, color=color[3], **kw)
    axes[1, 0].set_title("Latency per Batch")
    axes[1, 0].set_xlabel("Batch size")
    axes[1, 0].set_ylabel("ms")

    # Pareto front: throughput vs GPU memory
    ax = axes[1, 1]
    ax.scatter(gpu_mems, throughputs, color=color[2], s=80, zorder=5)
    for bs, x, y in zip(batch_sizes, gpu_mems, throughputs):
        ax.annotate(f"bs={bs}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    pareto = _pareto_front(gpu_mems, throughputs)
    px, py = zip(*pareto)
    ax.plot(px, py, linestyle="--", color=color[2], linewidth=1.5, alpha=0.6, label="Pareto front")
    ax.set_title("Throughput vs GPU Memory")
    ax.set_xlabel("Peak GPU Memory (MB)")
    ax.set_ylabel("Throughput (img/s)")
    ax.legend()

    for ax in axes.flat[:3]:
        ax.set_xticks(batch_sizes)
        ax.grid(True, alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "batch_size_study.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


def _pareto_front(memory, throughput):
    """Return (memory, throughput) points on the Pareto front (min memory, max throughput)."""
    points = sorted(zip(memory, throughput), key=lambda p: p[0])
    front, best_tput = [], float("-inf")
    for mem, tput in points:
        if tput > best_tput:
            front.append((mem, tput))
            best_tput = tput
    return front


if __name__ == "__main__":
    main()
