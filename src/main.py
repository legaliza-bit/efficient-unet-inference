import argparse
import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.config import DEVICE, BATCH_SIZE, DATA_DIR, RESULTS_DIR
from src.utils import set_seed
from src.data import download_carvana, get_carvana
from src.finetune.finetune import finetune, finetune_qat
from src.model import load_model, apply_compiled, apply_fp8, apply_int8
from src.benchmark import run_benchmark


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

    if (DATA_DIR / "train").exists():
        logger.info(f"Found existing dataset at {DATA_DIR}")
    elif args.download:
        download_carvana()
    else:
        raise FileNotFoundError(
            f"No dataset at {DATA_DIR}. Run with --download to fetch from Kaggle."
        )

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    model = load_model()

    if args.finetune:
        logger.info("\nFinetuning")
        finetune(model)
        model = load_model()

    if args.finetune_qat:
        logger.info("\nQuantization Aware Finetuning")
        finetune_qat(model)

    val_ds = get_carvana("val")
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=set_seed,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text("[]")

    # ── 1. FP16 baseline (GPU) ────────────────────────────────────
    run_benchmark(model, val_loader, DEVICE, "fp16_baseline", "fp32", args.profile)
    model.cpu()

    # # ── 2. torch.compile FP16 (GPU) ──────────────────────────────
    print("\nCompiling model (first run will be slow)…")
    run_benchmark(apply_compiled(model), val_loader, DEVICE, "compile_fp16", "fp16")

    # # ── 3. FP8 torchao (GPU, SM 8.9+) ───────────────────────────
    print("\nApplying FP8 weight quantization (torchao)…")
    run_benchmark(apply_fp8(model), val_loader, DEVICE, "fp8_torchao", "fp8")

    # # ── 4. torchao INT8 (GPU) ────────────────────────────────────
    run_benchmark(apply_int8(model), val_loader, DEVICE, "int8_torchao", "int8")

    # # ── 5 & 6. TRT experiments (GPU) ─────────────────────────────
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
        run_benchmark(TRTModel(TRT_FP16_PATH, DEVICE), val_loader, DEVICE, "trt_fp16", "fp16")

        print("\nBuilding TRT INT8 engine (calibrating)…")
        build_trt_engine(
            ONNX_PATH, TRT_INT8_PATH,
            int8=True, calib_path=CALIB_PATH,
        )
        run_benchmark(TRTModel(TRT_INT8_PATH, DEVICE), val_loader, DEVICE, "trt_int8", "int8")

    summary_path = RESULTS_DIR / "summary.json"
    print(f"\nResults saved to {summary_path}\n")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
