import time
import torch
from loguru import logger
import numpy as np

from src.utils import BenchmarkResult, reset_gpu_state, warmup_model, get_model_size_mb


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
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if use_fp16:
                x = x.half()

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
                pred = torch.argmax(out, dim=1)
                lat = (time.perf_counter() - t0) * 1000

            latencies.append(lat)

            mask = (y >= 0) & (y < num_classes) & (y != 255)

            conf_matrix += torch.bincount(
                (num_classes * y[mask] + pred[mask]).view(-1),
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    arr = np.array(latencies)
    total_time_sec = arr.sum() / 1000.0

    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection

    iou = intersection / (union + 1e-6)
    miou = iou.mean().item()

    dice = (2 * intersection) / (conf_matrix.sum(1) + conf_matrix.sum(0) + 1e-6)
    mean_dice = dice.mean().item()

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
