"""TVM integration module — launches the TVM benchmark as a subprocess.

TVM requires Python 3.11 (.venv-tvm311), while the main project uses
Python 3.13. This module provides the same interface as src/trt.py but
delegates all TVM work to a subprocess running under Python 3.11.
"""

import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from src.config import TMP_DIR

# Paths for the TVM venv and source build
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TVM_VENV_PYTHON = _PROJECT_ROOT / ".venv-tvm311" / "bin" / "python"
_TVM_BENCHMARK_SCRIPT = Path(__file__).resolve().parent / "_tvm_benchmark.py"

# Environment variables needed for TVM
_TVM_SRC = _PROJECT_ROOT / "tmp" / "tvm-src"
_LLVM_DIR = _PROJECT_ROOT / "tmp" / "clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04"
_LOCAL_LIB = _PROJECT_ROOT / "tmp" / "local-lib"

_TVM_ENV = {
    "PYTHONPATH": f"{_TVM_SRC / 'python'}",
    "LD_LIBRARY_PATH": (
        f"{_TVM_SRC / 'build'}:{_LLVM_DIR / 'lib'}:{_LOCAL_LIB}"
        f":/usr/lib/x86_64-linux-gnu"
    ),
}


def run_tvm_benchmark(
    onnx_path: Path,
    data_dir: Path,
    precision: str = "fp16",
    batch_size: int = 8,
    img_scale: float = 0.5,
    num_workers: int = 4,
    max_batches: int = 10,
    tune: bool = False,
    tune_trials: int = 1000,
) -> dict:
    """Run TVM benchmark as a subprocess under Python 3.11.

    Args:
        max_batches: Max batches to benchmark. 0 = all batches (very slow).
            Default is 10 for fast iteration.
        tune: If True, run AutoTVM tuning before compilation.

    Returns a dict with benchmark results matching BenchmarkResult fields.
    """
    # Pre-flight validation
    if not _TVM_VENV_PYTHON.exists():
        raise FileNotFoundError(
            f"TVM venv Python not found at {_TVM_VENV_PYTHON}. "
            f"Create the TVM venv following TVM_SETUP.md."
        )
    if not (_TVM_SRC / "python").exists():
        raise FileNotFoundError(
            f"TVM source build not found at {_TVM_SRC}. "
            f"Build TVM from source following TVM_SETUP.md."
        )

    output_dir = TMP_DIR / "tvm_results"

    cmd = [
        str(_TVM_VENV_PYTHON),
        str(_TVM_BENCHMARK_SCRIPT),
        "--onnx-path", str(onnx_path),
        "--precision", precision,
        "--output-dir", str(output_dir),
        "--batch-size", str(batch_size),
        "--data-dir", str(data_dir),
        "--img-scale", str(img_scale),
        "--num-workers", str(num_workers),
        "--max-batches", str(max_batches),
    ]

    if tune:
        cmd.append("--tune")
    if tune_trials != 1000:
        cmd.extend(["--tune-trials", str(tune_trials)])

    import os
    env = os.environ.copy()
    env.update(_TVM_ENV)

    logger.info(f"Launching TVM benchmark subprocess: {precision}"
                f" (max_batches={max_batches}, tune={tune})")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=False,  # Stream output for progress
        text=True,
        check=True,
    )

    # Read results from JSON
    results_path = output_dir / f"tvm_{precision}_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"TVM results not found at {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    logger.info(f"TVM benchmark complete: compute={results['latency_compute_mean_ms']:.2f} ms, "
                f"e2e={results['latency_e2e_mean_ms']:.2f} ms, "
                f"throughput={results['throughput_compute_samples_per_s']:.1f} samples/s")
    return results