#!/usr/bin/env python3
"""Standalone TVM benchmark script — runs under Python 3.11 (.venv-tvm311).

This script is launched as a subprocess by src/tvm.py because TVM is only
available in the Python 3.11 virtual environment, while the main project
uses Python 3.13.

Usage:
    python src/_tvm_benchmark.py \
        --onnx-path tmp/unet_carvana.onnx \
        --precision fp16 \
        --output-dir tmp/tvm_results \
        --batch-size 8 \
        --data-dir data/carvana \
        --img-scale 0.5 \
        --num-workers 4 \
        --max-batches 10 \
        --tune
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Set TVM environment before importing
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TVM_SRC = _PROJECT_ROOT / "tmp" / "tvm-src"
_LLVM_DIR = _PROJECT_ROOT / "tmp" / "clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04"
_LOCAL_LIB = _PROJECT_ROOT / "tmp" / "local-lib"

os.environ["PYTHONPATH"] = f"{_TVM_SRC / 'python'}:{os.environ.get('PYTHONPATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_TVM_SRC / 'build'}:{_LLVM_DIR / 'lib'}:{_LOCAL_LIB}"
    f":/usr/lib/x86_64-linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

# Initialize CUDA driver API before TVM uses it
import ctypes

try:
    ctypes.CDLL("libcuda.so.1").cuInit(0)
except Exception:
    pass  # May already be initialized

# Fix: src/tvm.py shadows the tvm package. Ensure TVM source is found first.
_tvm_python = str(_TVM_SRC / "python")
if _tvm_python not in sys.path:
    sys.path.insert(0, _tvm_python)
# Remove the script's directory from sys.path to avoid src/tvm.py shadowing
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import numpy as np
import torch
from torch.utils.data import Dataset
from tvm.contrib import graph_executor

import tvm
from tvm import relay

# TVM 0.12.0 compat: Sequential moved from relay.transform to tvm.transform
from tvm import transform as _tvm_transform

if not hasattr(relay.transform, "Sequential"):
    relay.transform.Sequential = _tvm_transform.Sequential


def _target_runtime_libs():
    """Return CUDA contrib libraries available in the current TVM build."""
    libs = []
    if tvm.get_global_func("tvm.contrib.cudnn.conv2d.forward", allow_missing=True) is not None:
        libs.append("cudnn")
    if tvm.get_global_func("tvm.contrib.cublas.matmul", allow_missing=True) is not None:
        libs.append("cublas")
    return libs


def _make_cuda_target():
    """Build the CUDA target with vendor libs when TVM was compiled with them."""
    dev = tvm.cuda(0)
    cc = dev.compute_version  # e.g. "9.0"
    sm_arch = "sm_" + cc.replace(".", "")  # e.g. "sm_90"
    libs = _target_runtime_libs()
    target_parts = [f"cuda -arch={sm_arch}"]
    if libs:
        target_parts.append(f"-libs={','.join(libs)}")
    target = tvm.target.Target(" ".join(target_parts))
    return target, sm_arch, libs


def _artifact_prefix(onnx_path, precision, batch_size, target, tune):
    """Return a stable cache prefix for compiled artifacts."""
    onnx_path = Path(onnx_path)
    stat = onnx_path.stat()
    payload = {
        "onnx": str(onnx_path.resolve()),
        "onnx_mtime_ns": stat.st_mtime_ns,
        "onnx_size": stat.st_size,
        "precision": precision,
        "batch_size": batch_size,
        "target": str(target),
        "tune": tune,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = _PROJECT_ROOT / "tmp" / "tvm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{onnx_path.stem}_{precision}_bs{batch_size}_{digest}"


def _load_compiled_artifact(cache_prefix):
    """Load a cached graph executor artifact if it exists."""
    lib_path = cache_prefix.with_suffix(".tar")
    graph_path = cache_prefix.with_suffix(".graph.json")
    params_path = cache_prefix.with_suffix(".params")
    if not (lib_path.exists() and graph_path.exists() and params_path.exists()):
        return None

    print(f"  Loading cached TVM artifact: {lib_path.name}")
    return {
        "lib": tvm.runtime.load_module(str(lib_path)),
        "graph_json": graph_path.read_text(),
        "params": bytearray(params_path.read_bytes()),
    }


def _save_compiled_artifact(factory_module, cache_prefix, metadata):
    """Persist a compiled graph executor factory to disk for reuse."""
    lib_path = cache_prefix.with_suffix(".tar")
    graph_path = cache_prefix.with_suffix(".graph.json")
    params_path = cache_prefix.with_suffix(".params")
    meta_path = cache_prefix.with_suffix(".meta.json")

    factory_module.get_lib().export_library(str(lib_path))
    graph_path.write_text(factory_module.get_graph_json())
    params_path.write_bytes(relay.save_param_dict(factory_module.get_params()))
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {
        "lib": tvm.runtime.load_module(str(lib_path)),
        "graph_json": graph_path.read_text(),
        "params": bytearray(params_path.read_bytes()),
    }


def _create_graph_module(compiled_module, dev):
    if isinstance(compiled_module, dict):
        module = graph_executor.create(
            compiled_module["graph_json"],
            compiled_module["lib"],
            dev,
        )
        module.load_params(compiled_module["params"])
        return module
    return graph_executor.GraphModule(compiled_module["default"](dev))


# ─── Dataset (self-contained, mirrors src/data.py) ─────────────────────

class CachedDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.imgs = data["imgs"]  # float16
        self.masks = data["masks"]  # uint8

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx].float(), self.masks[idx].long()

def get_dataloader(cache_path, batch_size, num_workers):
    """Create validation DataLoader from cache."""
    val_ds = CachedDataset(cache_path)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=num_workers, pin_memory=True,
    )
    return loader


# ─── AutoTVM Tuning ───────────────────────────────────────────────────

def _task_workload_key(task):
    return str(task.workload)


def _impact_score(task):
    """Rough FLOP-like score to tune heavier conv workloads first."""
    try:
        data_shape = task.args[0][1]
        kernel_shape = task.args[1][1]
        if len(data_shape) == 4 and len(kernel_shape) == 4:
            return int(np.prod(data_shape) * np.prod(kernel_shape[-2:]))
    except Exception:
        pass
    return 0


def _load_tuned_workloads(log_file):
    from tvm.autotvm.record import load_from_file

    workloads = set()
    if not log_file.exists() or log_file.stat().st_size == 0:
        return workloads

    try:
        for inp, _ in load_from_file(str(log_file)):
            workloads.add(str(inp.task.workload))
    except Exception:
        return set()
    return workloads


def auto_tune(relay_mod, params, target, dev, log_file, tune_trials=1000):
    """Run AutoTVM tuning for the relay model.

    Results are cached in per-model log files. Partial logs are resumed rather
    than treated as complete, which avoids silently skipping tuning after an
    interrupted run.
    """
    from tvm.autotvm.record import pick_best

    from tvm import autotvm

    tasks = autotvm.task.extract_from_program(
        relay_mod["main"], target=target, params=params
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    best_log_file = log_file.with_name(log_file.stem + ".best" + log_file.suffix)

    if not tasks:
        print("  AutoTVM: no tunable tasks extracted, building without tuning logs")
        with tvm.transform.PassContext(opt_level=4):
            return relay.build(relay_mod, target=target, params=params)

    expected_workloads = {_task_workload_key(task) for task in tasks}
    cached_workloads = _load_tuned_workloads(best_log_file)
    if expected_workloads and expected_workloads.issubset(cached_workloads):
        print(f"  AutoTVM: using cached best log ({len(cached_workloads)}/{len(expected_workloads)} workloads)")
        with autotvm.apply_history_best(str(best_log_file)):
            with tvm.transform.PassContext(opt_level=4):
                return relay.build(relay_mod, target=target, params=params)

    remaining_tasks = [task for task in tasks if _task_workload_key(task) not in cached_workloads]
    remaining_tasks.sort(key=_impact_score, reverse=True)

    print(f"  AutoTVM: extracted {len(tasks)} tasks, tuning {len(remaining_tasks)} missing workloads")
    print(f"  AutoTVM: best log: {best_log_file}")

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=60, n_parallel=1),
        runner=autotvm.LocalRunner(number=3, repeat=1, min_repeat_ms=100, timeout=20),
    )

    impactful_kinds = ("conv2d", "winograd", "conv2d_transpose", "dense", "matmul")
    for i, task in enumerate(remaining_tasks, start=1):
        tuner = autotvm.tuner.XGBTuner(task)
        task_name = str(task)

        if any(kind in task_name for kind in impactful_kinds):
            n_trial = min(tune_trials, len(task.config_space))
        else:
            n_trial = min(8, len(task.config_space))

        print(f"    Task {i}/{len(remaining_tasks)}: {task_name} — {n_trial} trials")
        tuner.tune(
            n_trial=n_trial,
            early_stopping=min(32, n_trial),
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(str(log_file))],
        )

    if log_file.exists() and log_file.stat().st_size > 0:
        pick_best(str(log_file), str(best_log_file))

    # Build with tuning logs
    print("  Building with AutoTVM best logs...")
    with autotvm.apply_history_best(str(best_log_file)):
        with tvm.transform.PassContext(opt_level=4):
            lib = relay.build(relay_mod, target=target, params=params)
    return lib


# ─── TVM Compilation ──────────────────────────────────────────────────

def compile_tvm_model(onnx_path, precision="fp16", batch_size=8, tune=False, tune_trials=1000):
    """Compile ONNX model to TVM graph executor module.

    Args:
        tune: If True, run AutoTVM tuning before compilation.
    """
    import onnx as onnx_lib
    onnx_model = onnx_lib.load(onnx_path)

    # Get input shape from ONNX model
    input_name = onnx_model.graph.input[0].name
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    # Handle dynamic batch
    if input_shape[0] == 0 or input_shape[0] == -1:
        input_shape[0] = batch_size

    shape_dict = {input_name: input_shape}

    print(f"  Importing ONNX → Relay (input: {input_name} {input_shape})...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict, freeze_params=True)

    # Apply FP16 conversion if requested
    if precision == "fp16":
        print("  Converting Relay graph to FP16...")
        # Mixed precision: keep loss-sensitive ops in FP32
        mod = relay.transform.ToMixedPrecision(
            mixed_precision_type="float16",
            missing_op_mode=1,  # 1 = skip ops that don't support FP16 (keep FP32)
        )(mod)

    # Optimize
    print("  Optimizing Relay graph...")
    seq = tvm.transform.Sequential([
        relay.transform.SimplifyInference(),
        relay.transform.FoldConstant(),
        relay.transform.FuseOps(),
        relay.transform.CombineParallelConv2D(),
    ])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    # Compile for CUDA with vendor libs when the local TVM build supports them.
    dev = tvm.cuda(0)
    target, sm_arch, target_libs = _make_cuda_target()
    print(f"  Target: {target} (detected {sm_arch})")
    if target_libs:
        print(f"  Using TVM contrib libs: {', '.join(target_libs)}")
    else:
        print("  TVM build has no cuDNN/cuBLAS integration; falling back to generic CUDA kernels")

    cache_prefix = _artifact_prefix(onnx_path, precision, batch_size, target, tune)
    cached_module = _load_compiled_artifact(cache_prefix)
    if cached_module is not None:
        return cached_module, input_name

    if tune:
        print("  Running AutoTVM tuning...")
        log_name = f"{Path(onnx_path).stem}_{precision}_{sm_arch}_bs{batch_size}.autotvm.json"
        log_file = _PROJECT_ROOT / "tmp" / "tvm_tuning_logs" / log_name
        factory_module = auto_tune(
            mod,
            params,
            target,
            dev,
            log_file=log_file,
            tune_trials=tune_trials,
        )
    else:
        print("  Compiling for CUDA...")
        with tvm.transform.PassContext(opt_level=4):
            factory_module = relay.build(mod, target=target, params=params)

    compiled_module = _save_compiled_artifact(
        factory_module,
        cache_prefix,
        metadata={
            "onnx_path": str(Path(onnx_path).resolve()),
            "precision": precision,
            "batch_size": batch_size,
            "target": str(target),
            "target_libs": target_libs,
            "sm_arch": sm_arch,
            "tuned": tune,
        },
    )

    return compiled_module, input_name


# ─── Quality Metrics ──────────────────────────────────────────────────

@torch.no_grad()
def update_conf_matrix(conf_matrix, preds, targets, num_classes):
    preds = preds.flatten()
    targets = targets.flatten()
    mask = targets < num_classes
    preds = preds[mask]
    targets = targets[mask]
    indices = targets * num_classes + preds
    conf_matrix.flatten().scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int64))


def compute_miou_dice(conf_matrix):
    tp = torch.diag(conf_matrix).float()
    fp = conf_matrix.sum(dim=0).float() - tp
    fn = conf_matrix.sum(dim=1).float() - tp
    iou = tp / (tp + fp + fn + 1e-8)
    miou = iou.mean().item()
    dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
    return miou, dice


# ─── Benchmark ────────────────────────────────────────────────────────

def run_benchmark(lib, input_name, dataloader, precision, num_classes=2, max_batches=0, tuned=False):
    """Run benchmark with TVM compiled module.

    Reports two timing measurements:
        - latency_compute_ms: only module.run() (comparable to PyTorch CUDA events)
        - latency_e2e_ms: set_input + run + get_output (realistic end-to-end with data transfers)

    Args:
        max_batches: Max number of batches to benchmark. 0 = run all batches (slow).
        tuned: Whether AutoTVM tuning was used for this build.
    """
    # Create GPU module
    dev = tvm.cuda(0)
    module = _create_graph_module(lib, dev)

    # Warmup
    print("  Warming up (20 iters)...")
    sample_input, _ = next(iter(dataloader))
    warmup_np = sample_input.numpy()
    for _ in range(20):
        module.set_input(input_name, tvm.nd.array(warmup_np, dev))
        module.run()

    # Timed inference
    print("  Running benchmark...")
    latencies_compute = []
    latencies_e2e = []
    total_samples = 0
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    peak_mem = 0
    dtype = sample_input.numpy().dtype

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            x_np = images.numpy().astype(dtype)
            batch_size = x_np.shape[0]

            # Set input BEFORE timing starts (H2D transfer not counted in compute time)
            module.set_input(input_name, tvm.nd.array(x_np, dev))

            # ── Compute-only timing: only module.run() ──
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            module.run()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_compute.append((t1 - t0) * 1000)  # ms

            # Get output AFTER compute timing ends (D2H transfer not counted in compute time)
            output = module.get_output(0).asnumpy()

            # ── End-to-end timing: set_input + run + get_output ──
            torch.cuda.synchronize()
            t_e2e_start = time.perf_counter()
            module.set_input(input_name, tvm.nd.array(x_np, dev))
            module.run()
            _ = module.get_output(0).asnumpy()
            torch.cuda.synchronize()
            t_e2e_end = time.perf_counter()
            latencies_e2e.append((t_e2e_end - t_e2e_start) * 1000)  # ms

            total_samples += batch_size

            # Quality metrics (computed on the output from the compute-only pass)
            pred = torch.from_numpy(output).argmax(dim=1)
            y_dev = masks
            update_conf_matrix(conf_matrix, pred, y_dev, num_classes)

    # Track GPU memory usage
    free_mem, total_mem = torch.cuda.mem_get_info(torch.cuda.current_device())
    used_mem = (total_mem - free_mem) / 1024**2  # MB currently used on GPU
    peak_mem = used_mem  # Approximation: current usage after benchmark

    # Compute stats for both timing modes
    arr_compute = np.array(latencies_compute)
    arr_e2e = np.array(latencies_e2e)

    # Trim outliers at p99 for compute latencies
    p99_compute = float(np.percentile(arr_compute, 99))
    arr_compute_trimmed = arr_compute[arr_compute <= p99_compute]

    # Trim outliers at p99 for e2e latencies
    p99_e2e = float(np.percentile(arr_e2e, 99))
    arr_e2e_trimmed = arr_e2e[arr_e2e <= p99_e2e]

    miou, mean_dice = compute_miou_dice(conf_matrix)

    num_batches = len(latencies_compute)

    results = {
        "pipeline_name": f"tvm_{precision}",
        "precision": precision,
        "device": "cuda:0",
        "batch_size": dataloader.batch_size,
        "num_batches": num_batches,
        "total_samples": total_samples,
        # Compute-only timing (only module.run(), comparable to PyTorch CUDA events)
        "latency_compute_mean_ms": float(arr_compute_trimmed.mean()),
        "latency_compute_std_ms": float(arr_compute_trimmed.std()),
        "latency_compute_p50_ms": float(np.percentile(arr_compute, 50)),
        "latency_compute_p95_ms": float(np.percentile(arr_compute, 95)),
        "latency_compute_p99_ms": float(np.percentile(arr_compute, 99)),
        "throughput_compute_samples_per_s": dataloader.batch_size * 1000.0 / float(arr_compute_trimmed.mean()),
        # End-to-end timing (set_input + run + get_output, includes H2D and D2H transfers)
        "latency_e2e_mean_ms": float(arr_e2e_trimmed.mean()),
        "latency_e2e_std_ms": float(arr_e2e_trimmed.std()),
        "latency_e2e_p50_ms": float(np.percentile(arr_e2e, 50)),
        "latency_e2e_p95_ms": float(np.percentile(arr_e2e, 95)),
        "latency_e2e_p99_ms": float(np.percentile(arr_e2e, 99)),
        "throughput_e2e_samples_per_s": dataloader.batch_size * 1000.0 / float(arr_e2e_trimmed.mean()),
        # Legacy keys for backward compatibility
        "latency_mean_ms": float(arr_compute_trimmed.mean()),
        "latency_std_ms": float(arr_compute_trimmed.std()),
        "latency_p50_ms": float(np.percentile(arr_compute, 50)),
        "latency_p95_ms": float(np.percentile(arr_compute, 95)),
        "latency_p99_ms": float(np.percentile(arr_compute, 99)),
        "throughput_samples_per_sec": dataloader.batch_size * 1000.0 / float(arr_compute_trimmed.mean()),
        # Quality metrics
        "miou": miou,
        "dice": mean_dice,
        "peak_gpu_memory_MB": peak_mem,
        "compile_time_s": None,
        "bench_time_s": None,
        "tuned": tuned,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="TVM benchmark (runs under Python 3.11)")
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Max batches to benchmark (0 = all batches, very slow)")
    parser.add_argument("--tune", action="store_true",
                        help="Run AutoTVM tuning before compilation (slow, but improves performance)")
    parser.add_argument("--tune-trials", type=int, default=1000,
                        help="Number of AutoTVM trials per task (default: 1000). "
                             "Only used with --tune.")
    args = parser.parse_args()

    print(f"╔══ TVM Benchmark ({args.precision}) ══╗")
    print(f"  ONNX: {args.onnx_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max batches: {args.max_batches if args.max_batches > 0 else 'all'}")
    print(f"  AutoTVM tuning: {args.tune}")
    print(f"  TVM version: {tvm.__version__}")
    print(f"  CUDA device: {tvm.cuda(0).device_name}")

    # Step 1: Compile
    print("\n── Compiling ──")
    t_compile_start = time.time()
    lib, input_name = compile_tvm_model(args.onnx_path, args.precision, args.batch_size,
                                         tune=args.tune, tune_trials=args.tune_trials)
    t_compile = time.time() - t_compile_start
    print(f"  Compilation done in {t_compile:.1f}s")

    # Step 2: Load data
    print("\n── Loading data ──")
    loader = get_dataloader(args.cache_path, args.batch_size, args.num_workers)
    print(f"  Dataset size: {len(loader.dataset)}, batches: {len(loader)}")

    # Step 3: Run benchmark
    print("\n── Benchmarking ──")
    t_bench_start = time.time()
    results = run_benchmark(lib, input_name, loader, args.precision,
                            max_batches=args.max_batches, tuned=args.tune)
    t_bench = time.time() - t_bench_start
    print(f"  Benchmark done in {t_bench:.1f}s")
    results["compile_time_s"] = t_compile
    results["bench_time_s"] = t_bench

    # Step 4: Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"tvm_{args.precision}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print(f"  Compute latency: {results['latency_compute_mean_ms']:.2f} ± {results['latency_compute_std_ms']:.2f} ms")
    print(f"  E2E latency:     {results['latency_e2e_mean_ms']:.2f} ± {results['latency_e2e_std_ms']:.2f} ms")
    print(f"  Compute throughput: {results['throughput_compute_samples_per_s']:.1f} samples/s")
    print(f"  E2E throughput:     {results['throughput_e2e_samples_per_s']:.1f} samples/s")
    print(f"  mIoU: {results['miou']:.4f}, Dice: {results['dice']:.4f}")


if __name__ == "__main__":
    main()