import gc
import json
import zipfile
import torch
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

from src.config import CKPT_PATH
from src.model import build_model


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

    model_params_M: Optional[float] = None
    model_size_MB: Optional[float] = None

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
            f"Latency:     {self.latency_mean_ms:.2f} ± "
            f"{self.latency_std_ms:.2f} ms "
            f"(p50={self.latency_p50_ms:.2f}, p95={self.latency_p95_ms:.2f})",
            f"Throughput:  {self.throughput_samples_per_sec:.1f} samples/s",
            f"mIoU:        {self.miou}",
            f"Dice:        {self.dice}",
        ]
        return "\n".join(lines)


def load_model():
    model = build_model()
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        print(f"Loaded checkpoint: {CKPT_PATH}")
    else:
        print("No checkpoint found — using milesial/Pytorch-UNet pretrained weights.")
    return model


def warmup_model(
    model: Callable,
    dummy_input: torch.Tensor,
    n_iters: int = 10,
    device: Optional[torch.device] = None,
    use_fp16: bool = False,
) -> None:
    with torch.no_grad():
        for _ in range(n_iters):
            if use_fp16 and device and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    if device and device.type == "cuda":
        torch.cuda.synchronize(device)


def get_model_size_mb(model: torch.nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1024**2


def print_results(results):
    header = (
        f"{'Experiment':<22} {'Device':<6} "
        f"{'Lat(ms)':<10} {'Tput(img/s)':<13} "
        f"{'mIoU':<8} {'Size(MB)'}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.pipeline_name:<22} {r.device:<6} "
            f"{r.latency_mean_ms:<10.1f} "
            f"{r.throughput_samples_per_sec:<13.1f} "
            f"{r.miou:<8.4f} {r.model_size_MB:.1f}"
        )


def download_carvana(dest_dir: Path) -> None:
    """Download Carvana images and masks from Kaggle into dest_dir.

    Requires ~/.kaggle/kaggle.json and competition rules accepted at
    kaggle.com/c/carvana-image-masking-challenge.
    Result layout: dest_dir/imgs/  and  dest_dir/masks/
    """
    import subprocess

    dest_dir.mkdir(parents=True, exist_ok=True)
    comp = "carvana-image-masking-challenge"

    for filename, folder_name, label in [
        ("train.zip",       "train",       "imgs"),
        ("train_masks.zip", "train_masks", "masks"),
    ]:
        print(f"Downloading {filename} …")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-f", filename, "-p", str(dest_dir)],
            check=True,
        )

        zip_path = dest_dir / filename
        print(f"Extracting {filename} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()

        extracted = dest_dir / folder_name
        target = dest_dir / label
        if extracted.exists() and not target.exists():
            extracted.rename(target)

    print("Carvana dataset ready.")


def reset_gpu_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
