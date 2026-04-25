import time
import torch
from torch.profiler import ProfilerActivity
from loguru import logger
import numpy as np

from src.utils import (
    BenchmarkResult, reset_gpu_state, warmup_model, get_model_size_mb,
)
from src.metrics import update_conf_matrix, compute_miou_dice
from src.config import NUM_CLASSES, PROFILE_DIR


def _forward(model, x: torch.Tensor, precision: str) -> torch.Tensor:
    if precision == "fp16" and x.device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return model(x)
    return model(x)


def _profile(model, dataloader, device, pipeline_name: str, precision: str) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    sched = torch.profiler.schedule(wait=1, warmup=1, active=3)
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for i, (x, _) in enumerate(dataloader):
            if i >= 5:
                break
            with torch.no_grad():
                _forward(model, x.to(device), precision)
            prof.step()

    trace_path = PROFILE_DIR / f"{pipeline_name}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"  Profile trace → {trace_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_benchmark(
    model,
    dataloader,
    device,
    pipeline_name: str,
    precision: str = "fp16",
    num_classes: int = NUM_CLASSES,
    profile: bool = False,
    skip_batches: int = 0,
) -> BenchmarkResult:
    model = model.to(device)
    model.eval()

    reset_gpu_state()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dummy_input = next(iter(dataloader))[0].to(device)
    warmup_model(model, dummy_input, n_iters=20, device=device, precision=precision)

    if profile:
        _profile(model, dataloader, device, pipeline_name, precision)

    latencies = []
    total_samples = 0
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            if batch_idx < skip_batches:
                # Warmup pass — run forward but don't record latency.
                out = _forward(model, x, precision)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            else:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    start_event.record()
                    out = _forward(model, x, precision)
                    end_event.record()
                    torch.cuda.synchronize()
                    lat = start_event.elapsed_time(end_event)
                else:
                    t0 = time.perf_counter()
                    out = _forward(model, x, precision)
                    lat = (time.perf_counter() - t0) * 1000

                latencies.append(lat)
                total_samples += batch_size

            pred = torch.argmax(out, dim=1)
            update_conf_matrix(conf_matrix, pred, y, num_classes)

    arr = np.array(latencies)
    p99 = float(np.percentile(arr, 99))
    arr_trimmed = arr[arr <= p99]
    miou, mean_dice = compute_miou_dice(conf_matrix)
    peak_mem = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda" else None
    )

    result = BenchmarkResult(
        pipeline_name=pipeline_name,
        precision=precision,
        device=str(device),
        batch_size=dataloader.batch_size,
        num_batches=len(dataloader),
        total_samples=total_samples,
        latency_mean_ms=float(arr_trimmed.mean()),
        latency_std_ms=float(arr_trimmed.std()),
        latency_p50_ms=float(np.percentile(arr, 50)),
        latency_p95_ms=float(np.percentile(arr, 95)),
        latency_p99_ms=float(np.percentile(arr, 99)),
        throughput_samples_per_sec=dataloader.batch_size * 1000.0 / float(arr_trimmed.mean()),
        model_params_M=sum(p.numel() for p in model.parameters()) / 1e6,
        model_size_MB=get_model_size_mb(model),
        peak_gpu_memory_MB=peak_mem,
        miou=miou,
        dice=mean_dice,
    )

    logger.info(result)
    return result
