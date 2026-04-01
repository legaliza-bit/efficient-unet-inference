import gc
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 21) -> float:
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            continue
        ious.append((inter / union).item())
    return sum(ious) / len(ious) if ious else 0.0


@dataclass
class BenchmarkResult:
    pipeline_name: str
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

    miou: Optional[float] = None
    model_params_M: Optional[float] = None
    model_size_MB: Optional[float] = None

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
        lines = [
            f"Pipeline:    {self.pipeline_name}",
            f"Device:      {self.device}",
            f"Batch size:  {self.batch_size}",
            f"Samples:     {self.total_samples}",
            f"Latency:     {self.latency_mean_ms:.2f} +/- {self.latency_std_ms:.2f} ms "
            f"(p50={self.latency_p50_ms:.2f}, p95={self.latency_p95_ms:.2f})",
            f"Throughput:  {self.throughput_samples_per_sec:.1f} samples/s",
        ]
        if self.miou is not None:
            lines.append(f"mIoU:        {self.miou:.4f}")
        return "\n".join(lines)


class LatencyTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self._latencies: list[float] = []
        if self.is_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)

    @contextmanager
    def measure(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            self._start.record()
            yield
            self._end.record()
            torch.cuda.synchronize(self.device)
            self._latencies.append(self._start.elapsed_time(self._end))
        else:
            t0 = time.perf_counter()
            yield
            self._latencies.append((time.perf_counter() - t0) * 1000.0)

    def reset(self):
        self._latencies.clear()

    def stats(self) -> dict[str, float]:
        arr = np.array(self._latencies)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "total_ms": float(np.sum(arr)),
        }


def warmup_model(model, dummy_input, n_iters=10, device=None):
    with torch.no_grad():
        for _ in range(n_iters):
            model(dummy_input)
    if device and device.type == "cuda":
        torch.cuda.synchronize(device)


def get_model_size_mb(model: torch.nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1024**2


def reset_gpu_state():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
