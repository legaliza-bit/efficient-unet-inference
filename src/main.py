import argparse
import os

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config import DEVICE, BATCH_SIZE, DATA_DIR
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
    bench(apply_compiled(model), val_loader, DEVICE, "compile_fp16", "fp16")

    # ── 3. FP8 torchao (GPU, SM 8.9+) ───────────────────────────
    print("\nApplying FP8 weight quantization (torchao)…")
    bench(apply_fp8(model), val_loader, DEVICE, "fp8_torchao", "fp8")

    # ── 4. torchao INT8 (GPU) ────────────────────────────────────
    bench(apply_int8(model), val_loader, DEVICE, "int8_torchao", "int8")

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

    print_results(results)

    if args.profile:
        from src.config import PROFILE_DIR
        print(f"\nChrome traces saved to {PROFILE_DIR}/")
        print("Open at chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
