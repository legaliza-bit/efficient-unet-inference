import gc
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import torch
import numpy as np


@dataclass
class BenchmarkResult:
    pipeline_name: str
    device: str
    batch_size: int
    num_batches: int
    total_samples: int

    # Latency
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float

    # Throughput
    throughput_samples_per_sec: float

    # Model info
    model_params_M: Optional[float] = None
    model_size_MB: Optional[float] = None

    # Model quality
    miou: Optional[float] = None
    dice: Optional[float] = None

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
            f"Latency:     {self.latency_mean_ms:.2f} ± {self.latency_std_ms:.2f} ms "
            f"(p50={self.latency_p50_ms:.2f}, p95={self.latency_p95_ms:.2f})",
            f"Throughput:  {self.throughput_samples_per_sec:.1f} samples/s",
        ]
        return "\n".join(lines)


class LatencyTimer:
    """
    Точный таймер для GPU/CPU инференса.
    Использует CUDA Events для GPU, time.perf_counter для CPU.
    """

    def __init__(self, device: torch.device, warmup_iters: int = 10):
        self.device = device
        self.warmup_iters = warmup_iters
        self.is_cuda = device.type == "cuda"
        self._latencies: list[float] = []

        if self.is_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    @contextmanager
    def measure(self):
        """Контекстный менеджер для измерения одной итерации."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            self._start_event.record()
            yield
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            self._latencies.append(self._start_event.elapsed_time(self._end_event))
        else:
            t0 = time.perf_counter()
            yield
            self._latencies.append((time.perf_counter() - t0) * 1000.0)

    def reset(self):
        self._latencies.clear()

    def stats(self) -> dict[str, float]:
        if not self._latencies:
            raise RuntimeError("Нет замеров!")
        arr = np.array(self._latencies)
        return {
            "mean_ms":  float(np.mean(arr)),
            "std_ms":   float(np.std(arr)),
            "p50_ms":   float(np.percentile(arr, 50)),
            "p95_ms":   float(np.percentile(arr, 95)),
            "p99_ms":   float(np.percentile(arr, 99)),
            "min_ms":   float(np.min(arr)),
            "max_ms":   float(np.max(arr)),
            "total_ms": float(np.sum(arr)),
        }


def warmup_model(
    model: Callable,
    dummy_input: torch.Tensor,
    n_iters: int = 10,
    device: Optional[torch.device] = None,
) -> None:
    """Прогрев модели для стабилизации GPU-состояния."""
    with torch.no_grad():
        for _ in range(n_iters):
            with torch.amp.autocast("cuda"):
                _ = model(dummy_input)
    if device and device.type == "cuda":
        torch.cuda.synchronize(device)


def get_gpu_memory_mb() -> float:
    """Текущее потребление GPU памяти в МБ."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2
    return 0.0


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Размер весов модели в МБ."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1024**2


def reset_gpu_state() -> None:
    """Очистка кэша GPU перед бенчмарком."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
