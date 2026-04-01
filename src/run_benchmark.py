import time

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.config import NUM_CLASSES, WARMUP_ITERS
from src.utils import (
    BenchmarkResult,
    get_model_size_mb,
    miou,
    reset_gpu_state,
    warmup_model,
)


def run_benchmark(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pipeline_name: str,
    num_classes: int = NUM_CLASSES,
) -> BenchmarkResult:
    model = model.to(device)
    model.eval()

    use_fp16 = device.type == "cuda"
    reset_gpu_state()

    # warmup
    dummy = next(iter(dataloader))[0].to(device)
    if use_fp16:
        dummy = dummy.half()
    warmup_model(model, dummy, n_iters=WARMUP_ITERS, device=device)

    latencies = []
    total_samples = 0
    miou_acc = 0.0

    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            if use_fp16:
                x = x.half()

            if device.type == "cuda":
                torch.cuda.synchronize()
                start_ev.record()
                out = model(x)
                end_ev.record()
                torch.cuda.synchronize()
                lat = start_ev.elapsed_time(end_ev)
            else:
                t0 = time.perf_counter()
                out = model(x)
                lat = (time.perf_counter() - t0) * 1000.0

            latencies.append(lat)
            miou_acc += miou(out.float(), y, num_classes)
            total_samples += x.size(0)

    arr = np.array(latencies)
    total_sec = arr.sum() / 1000.0
    n_batches = len(latencies)
    avg_miou = miou_acc / n_batches if n_batches else 0.0

    result = BenchmarkResult(
        pipeline_name=pipeline_name,
        device=str(device),
        batch_size=dataloader.batch_size or 1,
        num_batches=n_batches,
        total_samples=total_samples,
        latency_mean_ms=float(arr.mean()),
        latency_std_ms=float(arr.std()),
        latency_p50_ms=float(np.percentile(arr, 50)),
        latency_p95_ms=float(np.percentile(arr, 95)),
        latency_p99_ms=float(np.percentile(arr, 99)),
        throughput_samples_per_sec=total_samples / total_sec if total_sec > 0 else 0.0,
        miou=avg_miou,
        model_params_M=sum(p.numel() for p in model.parameters()) / 1e6,
        model_size_MB=get_model_size_mb(model),
    )

    logger.info(f"\n{result}")
    return result
