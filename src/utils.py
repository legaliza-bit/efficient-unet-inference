import gc
import torch
import numpy as np
import random

from loguru import logger
from pathlib import Path


def set_seed():
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_model_size_mb(model) -> float:
    if hasattr(model, "model_path"):
        return Path(model.model_path).stat().st_size / 1024**2
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1024**2


def print_results(results):
    header = (
        f"{'Experiment':<22} {'Prec':<6} {'Dev':<5} "
        f"{'Lat(ms)':<10} {'p95(ms)':<9} {'Tput(img/s)':<13} "
        f"{'mIoU':<8} {'Dice':<8} {'GPU mem(MB)':<13} {'Size(MB)'}"
    )
    print("\n" + header)
    print("─" * len(header))
    for r in results:
        gpu_mem = f"{r.peak_gpu_memory_MB:.0f}" if r.peak_gpu_memory_MB else "n/a"
        dev = r.device.split(":")[0]
        print(
            f"{r.pipeline_name:<22} {r.precision:<6} {dev:<5} "
            f"{r.latency_mean_ms:<10.1f} {r.latency_p95_ms:<9.1f} "
            f"{r.throughput_samples_per_sec:<13.1f} "
            f"{r.miou:<8.4f} {r.dice:<8.4f} {gpu_mem:<13} {r.model_size_MB:.1f}"
        )


def log_gpu(tag: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        free, total = torch.cuda.mem_get_info()
        free = free / 1024**2
        total = total / 1024**2

        logger.info(
            f"[GPU {tag}] "
            f"allocated={alloc:.1f}MB | reserved={reserv:.1f}MB | "
            f"free={free:.1f}MB | total={total:.1f}MB"
        )


def reset_gpu_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    torch._dynamo.reset()
