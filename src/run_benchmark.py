import time
import torch
from loguru import logger
import numpy as np

from src.utils import (
    BenchmarkResult, reset_gpu_state, warmup_model, get_model_size_mb,
)
from src.metrics import update_conf_matrix, compute_miou_dice
from src.config import NUM_CLASSES


def run_benchmark(
    model,
    dataloader,
    device,
    pipeline_name: str,
    num_classes: int = NUM_CLASSES,
    use_fp16: bool = False,
):
    model = model.to(device)
    model.eval()

    reset_gpu_state()

    dummy_input = next(iter(dataloader))[0].to(device)
    warmup_model(
        model, dummy_input, n_iters=20, device=device, use_fp16=use_fp16
    )

    latencies = []
    total_samples = 0
    conf_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event.record()
                if use_fp16:
                    with torch.amp.autocast("cuda"):
                        out = model(x)
                else:
                    out = model(x)
                end_event.record()
                torch.cuda.synchronize()
                lat = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                out = model(x)
                lat = (time.perf_counter() - t0) * 1000

            pred = torch.argmax(out, dim=1)
            latencies.append(lat)
            total_samples += batch_size
            update_conf_matrix(conf_matrix, pred, y, num_classes)

    arr = np.array(latencies)
    total_time_sec = arr.sum() / 1000.0
    miou, mean_dice = compute_miou_dice(conf_matrix)

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
        model_params_M=sum(p.numel() for p in model.parameters()) / 1e6,
        model_size_MB=get_model_size_mb(model),
        miou=miou,
        dice=mean_dice,
    )

    logger.info(result)
    return result
