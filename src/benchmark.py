import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.profiler import ProfilerActivity

from src.config import NUM_CLASSES, PROFILE_DIR, RESULTS_DIR
from src.metrics import compute_miou_dice, update_conf_matrix
from src.model import forward, warmup_model
from src.utils import get_model_size_mb, reset_gpu_state


@dataclass
class BenchmarkResult:
    pipeline_name: str
    precision: str
    device: str
    batch_size: int
    num_batches: int
    total_samples: int

    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float

    throughput_samples_per_sec: float

    model_params_M: float | None = None
    model_size_MB: float | None = None
    peak_gpu_memory_MB: float | None = None

    miou: float | None = None
    dice: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path):
        with open(path) as f:
            return cls(**json.load(f))

    def __str__(self) -> str:
        mem = f"{self.peak_gpu_memory_MB:.0f} MB" if self.peak_gpu_memory_MB else "n/a"
        lines = [
            f"Pipeline:    {self.pipeline_name}",
            f"Precision:   {self.precision}",
            f"Device:      {self.device}",
            f"Batch size:  {self.batch_size}",
            f"Samples:     {self.total_samples}",
            f"Latency:     {self.latency_mean_ms:.2f} ± "
            f"{self.latency_std_ms:.2f} ms "
            f"(p50={self.latency_p50_ms:.2f}, p95={self.latency_p95_ms:.2f})",
            f"Throughput:  {self.throughput_samples_per_sec:.1f} samples/s",
            f"Peak GPU mem:{mem}",
            f"mIoU:        {self.miou}",
            f"Dice:        {self.dice}",
        ]
        return "\n".join(lines)


def profile(model, dataloader, device, pipeline_name: str, precision: str) -> None:
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
                forward(model, x.to(device), precision)
            prof.step()

    trace_path = PROFILE_DIR / f"{pipeline_name}.json"
    prof.export_chrome_trace(str(trace_path))
    logger.info(f"  Profile trace → {trace_path}")
    logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_benchmark(
    model,
    dataloader,
    device,
    pipeline_name: str,
    precision: str = "fp16",
    profile: bool = False,
) -> BenchmarkResult:
    model = model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dummy_input = next(iter(dataloader))[0].to(device)
    warmup_model(model, dummy_input, n_iters=20, device=device, precision=precision)

    if profile:
        profile(model, dataloader, device, pipeline_name, precision)

    latencies = []
    total_samples = 0
    conf_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device
    )

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event.record()
                out = forward(model, x, precision)
                end_event.record()
                torch.cuda.synchronize()
                lat = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                out = forward(model, x, precision)
                lat = (time.perf_counter() - t0) * 1000

            pred = torch.argmax(out, dim=1)
            latencies.append(lat)
            total_samples += batch_size
            update_conf_matrix(conf_matrix, pred, y, NUM_CLASSES)

    arr = np.array(latencies)
    total_time_sec = arr.sum() / 1000.0
    p99 = float(np.percentile(arr, 99))
    arr_trimmed = arr[arr <= p99]
    miou, mean_dice = compute_miou_dice(conf_matrix)
    peak_mem = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
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
        throughput_samples_per_sec=total_samples / total_time_sec,
        model_params_M=sum(p.numel() for p in model.parameters()) / 1e6,
        model_size_MB=get_model_size_mb(model),
        peak_gpu_memory_MB=peak_mem,
        miou=miou,
        dice=mean_dice,
    )
    logger.info(result)

    summary_path = RESULTS_DIR / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    existing.append(result.to_dict())
    summary_path.write_text(json.dumps(existing, indent=2))

    del model
    reset_gpu_state()

    return result
