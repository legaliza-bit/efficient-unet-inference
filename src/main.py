import argparse
import os

import torch
from torch.utils.data import DataLoader
from loguru import logger

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config import DEVICE, BATCH_SIZE, DATA_DIR, IMG_SCALE
from src.utils import load_model, print_results, download_carvana, reset_gpu_state, log_gpu
from src.data import get_carvana
from src.finetune.finetune import finetune, finetune_qat
from src.model import apply_compiled, apply_int8, apply_fp8
from src.run_benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune-qat", action="store_true")
    parser.add_argument("--download", action="store_true",
                        help="Download Carvana from Kaggle (requires ~/.kaggle/kaggle.json)")
    parser.add_argument("--trt", action="store_true",
                        help="Include TensorRT FP16 experiment (requires tensorrt-cu12)")
    parser.add_argument("--tvm", action="store_true",
                        help="Include TVM FP16/FP32 experiments (requires .venv-tvm311)")
    parser.add_argument("--tvm-tune", action="store_true",
                        help="Run AutoTVM tuning before TVM compilation (slow but improves performance)")
    parser.add_argument("--tvm-tune-trials", type=int, default=1000,
                        help="Number of AutoTVM trials per task (default: 1000). "
                             "Only used with --tvm --tvm-tune.")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler on each experiment and save chrome traces to tmp/profiles/")
    args = parser.parse_args()

    carvana_dir = DATA_DIR / "carvana"
    if (carvana_dir / "imgs").exists():
        print(f"Found existing dataset at {carvana_dir}")
    elif args.download:
        download_carvana(carvana_dir)
    else:
        raise FileNotFoundError(
            f"No dataset at {carvana_dir}. Run with --download to fetch from Kaggle."
        )

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm < 80:
            logger.warning("INT8 tensor cores require SM 80+ (Ampere). Performance may be limited.")
        if sm < 89:
            logger.warning("FP8 tensor cores require SM 89+ (Ada Lovelace/Hopper). FP8 will use emulation.")

    model = load_model()

    if args.finetune:
        print("\n── Finetuning ──────────────────────────────────")
        finetune(model)
        model = load_model()

    if args.finetune_qat:
        print("\n── QAT Finetuning ───────────────────────────────")
        finetune_qat(model)

    val_ds = get_carvana("val")
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"),
    )
    train_ds = get_carvana("train")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"),
    )
    results = []

    def bench(m, loader, device, name, precision, **kw):
        print(f"\n── {name} {'─' * max(0, 52 - len(name))}")
        log_gpu(f"start {name}")

        r = run_benchmark(m, loader, device, name, precision,
                          profile=args.profile, **kw)
        log_gpu(f"end {name}")

        results.append(r)
        del m
        reset_gpu_state()
        model.cpu()

    # ── 1. FP16 baseline (GPU) ────────────────────────────────────
    bench(model, val_loader, DEVICE, "fp16_baseline", "fp16")

    # ── 2. torch.compile FP16 (GPU) ──────────────────────────────
    print("\nCompiling model (first run will be slow)…")
    bench(apply_compiled(model), val_loader, DEVICE, "compile_fp16", "fp16",
          skip_batches=2)

    # ── 3. FP8 torchao (GPU, SM 8.9+) ───────────────────────────
    print("\nApplying FP8 dynamic activation + weight quantization (torchao)…")
    bench(torch.compile(apply_fp8(model), mode="max-autotune-no-cudagraphs"),
          val_loader, DEVICE, "fp8_torchao", "fp8", skip_batches=2)

    # ── 4. torchao INT8 (GPU) ────────────────────────────────────
    print("\nApplying INT8 static activation + weight quantization (torchao)…")
    bench(torch.compile(apply_int8(model, calib_dataloader=train_loader), mode="max-autotune-no-cudagraphs"),
          val_loader, DEVICE, "int8_torchao", "int8", skip_batches=2)

    # ── 5 & 6. TRT experiments (GPU) ─────────────────────────────
    if args.trt:
        try:
            import tensorrt as _trt  # noqa: F401 — availability check
        except ImportError:
            raise ImportError(
                "TensorRT is required for --trt. Install with: uv sync --extra trt"
            )
        from src.config import (
            ONNX_PATH, TRT_FP16_PATH, TRT_INT8_PATH, CALIB_PATH,
        )
        from src.trt import (
            export_to_onnx, save_calib_data, build_trt_engine, TRTModel,
        )
        sample, _ = next(iter(val_loader))
        export_to_onnx(model, sample, ONNX_PATH)
        save_calib_data(val_loader, n_samples=200, calib_path=CALIB_PATH)

        print("\nBuilding TRT FP16 engine…")
        build_trt_engine(ONNX_PATH, TRT_FP16_PATH, fp16=True)
        bench(TRTModel(TRT_FP16_PATH, DEVICE), val_loader, DEVICE, "trt_fp16", "fp16")

        print("\nBuilding TRT INT8 engine (calibrating)…")
        build_trt_engine(
            ONNX_PATH, TRT_INT8_PATH,
            int8=True, calib_path=CALIB_PATH,
        )
        bench(TRTModel(TRT_INT8_PATH, DEVICE), val_loader, DEVICE, "trt_int8", "int8")

    # ── TVM experiments
    if args.tvm:
        from src.config import ONNX_PATH
        from src.trt import export_to_onnx
        from src.tvm import run_tvm_benchmark

        # Ensure ONNX export exists
        if not ONNX_PATH.exists():
            sample, _ = next(iter(val_loader))
            export_to_onnx(model, sample, ONNX_PATH)

        carvana_dir = DATA_DIR / "carvana"

        # TVM FP16
        print("\n── TVM FP16 ──")
        tvm_fp16_results = run_tvm_benchmark(
            onnx_path=ONNX_PATH,
            data_dir=carvana_dir,
            precision="fp16",
            batch_size=BATCH_SIZE,
            img_scale=IMG_SCALE,
            max_batches=10,
            tune=args.tvm_tune,
            tune_trials=args.tvm_tune_trials,
        )
        # Convert to BenchmarkResult for consistent output
        from src.utils import BenchmarkResult
        tvm_fp16_result = BenchmarkResult(
            pipeline_name=tvm_fp16_results["pipeline_name"],
            precision=tvm_fp16_results["precision"],
            device=tvm_fp16_results["device"],
            batch_size=tvm_fp16_results["batch_size"],
            num_batches=tvm_fp16_results.get("num_batches", 0),
            total_samples=tvm_fp16_results["total_samples"],
            latency_mean_ms=tvm_fp16_results["latency_compute_mean_ms"],
            latency_std_ms=tvm_fp16_results["latency_compute_std_ms"],
            latency_p50_ms=tvm_fp16_results["latency_compute_p50_ms"],
            latency_p95_ms=tvm_fp16_results["latency_compute_p95_ms"],
            latency_p99_ms=tvm_fp16_results["latency_compute_p99_ms"],
            throughput_samples_per_sec=tvm_fp16_results["throughput_compute_samples_per_s"],
            miou=tvm_fp16_results.get("miou"),
            dice=tvm_fp16_results.get("dice"),
            peak_gpu_memory_MB=tvm_fp16_results.get("peak_gpu_memory_MB"),
        )
        results.append(tvm_fp16_result)
        print(f"  TVM FP16 compute: {tvm_fp16_results['latency_compute_mean_ms']:.2f} ms, "
              f"e2e: {tvm_fp16_results['latency_e2e_mean_ms']:.2f} ms")

        # TVM FP32
        print("\n── TVM FP32 ──")
        tvm_fp32_results = run_tvm_benchmark(
            onnx_path=ONNX_PATH,
            data_dir=carvana_dir,
            precision="fp32",
            batch_size=BATCH_SIZE,
            img_scale=IMG_SCALE,
            max_batches=10,
            tune=args.tvm_tune,
            tune_trials=args.tvm_tune_trials,
        )
        tvm_fp32_result = BenchmarkResult(
            pipeline_name=tvm_fp32_results["pipeline_name"],
            precision=tvm_fp32_results["precision"],
            device=tvm_fp32_results["device"],
            batch_size=tvm_fp32_results["batch_size"],
            num_batches=tvm_fp32_results.get("num_batches", 0),
            total_samples=tvm_fp32_results["total_samples"],
            latency_mean_ms=tvm_fp32_results["latency_compute_mean_ms"],
            latency_std_ms=tvm_fp32_results["latency_compute_std_ms"],
            latency_p50_ms=tvm_fp32_results["latency_compute_p50_ms"],
            latency_p95_ms=tvm_fp32_results["latency_compute_p95_ms"],
            latency_p99_ms=tvm_fp32_results["latency_compute_p99_ms"],
            throughput_samples_per_sec=tvm_fp32_results["throughput_compute_samples_per_s"],
            miou=tvm_fp32_results.get("miou"),
            dice=tvm_fp32_results.get("dice"),
            peak_gpu_memory_MB=tvm_fp32_results.get("peak_gpu_memory_MB"),
        )
        results.append(tvm_fp32_result)
        print(f"  TVM FP32 compute: {tvm_fp32_results['latency_compute_mean_ms']:.2f} ms, "
              f"e2e: {tvm_fp32_results['latency_e2e_mean_ms']:.2f} ms")

    print_results(results)

    if args.profile:
        from src.config import PROFILE_DIR
        print(f"\nChrome traces saved to {PROFILE_DIR}/")
        print("Open at chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
