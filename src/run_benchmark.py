import time
import torch
from loguru import logger
import numpy as np

from src.utils import BenchmarkResult, reset_gpu_state, warmup_model, get_model_size_mb, miou


def run_benchmark(model, dataloader, device, pipeline_name: str, num_classes: int = 21):
    model = model.to(device)
    model.eval()

    use_fp16 = device == "cuda"

    reset_gpu_state()

    dummy_input = next(iter(dataloader))[0].to(device)
    if use_fp16:
        dummy_input = dummy_input.half()

    warmup_model(model, dummy_input, n_iters=20, device=device)

    latencies = []
    total_samples = 0
    total_miou = 0.0

    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if use_fp16:
                x = x.half()

            batch_size = x.size(0)

            if device == "cuda":
                torch.cuda.synchronize()
                start_event.record()

                out = model(x)

                end_event.record()
                torch.cuda.synchronize()

                lat = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                out = model(x)
                lat = (time.perf_counter() - t0) * 1000

            latencies.append(lat)

            total_miou += miou(out, y, num_classes)

            total_samples += batch_size

    arr = np.array(latencies)
    total_time_sec = arr.sum() / 1000.0

    result = BenchmarkResult(
        pipeline_name=pipeline_name,
        device=str(device),
        batch_size=dataloader.batch_size,
        num_batches=len(dataloader),
        total_samples=total_samples,

        latency_mean_ms=float(arr.mean()),
        latency_std_ms=float(arr.std()),
        latency_p50_ms=float(np.percentile(arr, 50)),
        latency_p95_ms=float(np.percentile(arr, 95)),
        latency_p99_ms=float(np.percentile(arr, 99)),

        throughput_samples_per_sec=total_samples / total_time_sec,

        mean_iou=total_miou / len(dataloader),

        model_params_M=sum(p.numel() for p in model.parameters()) / 1e6,
        model_size_MB=get_model_size_mb(model),
    )

    logger.info(result)
    return result
