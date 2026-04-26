import argparse
import json
import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.config import DEVICE, BATCH_SIZE, DATA_DIR, RESULTS_DIR, CACHE_PATH, BENCH_N_SAMPLES
from src.utils import set_seed, print_results
from src.benchmark import run_benchmark, BenchmarkResult
from src.data import download_carvana, prepare_benchmark_cache, download_bench_cache, CachedDataset
from src.finetune.finetune import finetune, finetune_qat
from src.model import load_model, apply_compiled, apply_fp8, apply_int8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune-qat", action="store_true")
    parser.add_argument("--download", action="store_true",
                        help="Download Carvana from Kaggle (requires ~/.kaggle/kaggle.json)")
    parser.add_argument("--trt", action="store_true",
                        help="Include TensorRT experiments (requires tensorrt-cu12)")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler on each experiment and save chrome traces to tmp/profiles/")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[BATCH_SIZE],
                        metavar="BS", help="Batch sizes to sweep (e.g. --batch-sizes 1 4 8 16)")
    parser.add_argument("--prepare-cache", action="store_true",
                        help=f"Preprocess {BENCH_N_SAMPLES} samples from Carvana and save to tmp/bench_cache.pt")
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    model = load_model()

    # ── Prepare cache (needs raw Carvana data) ───────────────────────────────
    if args.prepare_cache:
        if args.download:
            download_carvana()
        elif not (DATA_DIR / "train").exists():
            raise FileNotFoundError(
                f"No dataset at {DATA_DIR}. Run with --prepare-cache --download to fetch from Kaggle and prepare cache."
            )
        if args.finetune:
            logger.info("\nFinetuning")
            finetune(model)
            model = load_model()
        if args.finetune_qat:
            logger.info("\nQuantization Aware Finetuning")
            finetune_qat(model)
        prepare_benchmark_cache()
        logger.info("Cache ready. Upload tmp/bench_cache.pt to Google Drive and set GDRIVE_FILE_ID in config.py")
        return

    # ── Benchmark (only needs cache) ─────────────────────────────────────────
    if CACHE_PATH.exists():
        logger.info(f"Loading benchmark cache from {CACHE_PATH}")
        dataset = CachedDataset(CACHE_PATH)
    else:
        logger.info("No cache found, attempting download from Google Drive...")
        download_bench_cache()
        dataset = CachedDataset(CACHE_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text("[]")

    # ── Build TRT engines once (dynamic batch, reused across BS sweep) ──────
    trt_models = {}
    if args.trt:
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorRT is required for --trt. Install with: uv sync --extra trt"
            )
        from src.config import (
            ONNX_PATH, TRT_FP16_PATH, TRT_FP8_PATH, TRT_INT8_PATH, CALIB_PATH,
        )
        from src.trt import export_to_onnx, save_calib_data, build_trt_engine, TRTModel

        calib_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=(DEVICE.type == "cuda"),
        )
        sample, _ = next(iter(calib_loader))
        export_to_onnx(model, sample, ONNX_PATH)
        save_calib_data(calib_loader, n_samples=200, calib_path=CALIB_PATH)

        print("\nBuilding TRT FP16 engine…")
        build_trt_engine(ONNX_PATH, TRT_FP16_PATH, fp16=True)
        trt_models["trt_fp16"] = ("fp16", TRT_FP16_PATH)

        print("\nBuilding TRT FP8 engine (calibrating)…")
        build_trt_engine(ONNX_PATH, TRT_FP8_PATH, fp8=True, calib_path=CALIB_PATH)
        trt_models["trt_fp8"] = ("fp8", TRT_FP8_PATH)

        print("\nBuilding TRT INT8 engine (calibrating)…")
        build_trt_engine(ONNX_PATH, TRT_INT8_PATH, int8=True, calib_path=CALIB_PATH)
        trt_models["trt_int8"] = ("int8", TRT_INT8_PATH)

    # ── Batch size sweep ─────────────────────────────────────────────────────
    for bs in args.batch_sizes:
        logger.info(f"Batch size: {bs}")

        dataloader = DataLoader(
            dataset, batch_size=bs, shuffle=False,
            num_workers=4, pin_memory=(DEVICE.type == "cuda"),
            worker_init_fn=set_seed,
        )
        run_benchmark(model, dataloader, DEVICE, f"fp32_baseline_bs{bs}", "fp32", args.profile)

        run_benchmark(model, dataloader, DEVICE, f"fp16_baseline_bs{bs}", "fp16", args.profile)

        run_benchmark(apply_compiled(model), dataloader, DEVICE, f"compile_fp16_bs{bs}", "fp16", args.profile)

        logger.info("\nApplying FP8 weight quantization (torchao)…")
        run_benchmark(apply_fp8(model), dataloader, DEVICE, f"fp8_torchao_bs{bs}", "fp8", args.profile)

        run_benchmark(apply_int8(model), dataloader, DEVICE, f"int8_torchao_bs{bs}", "int8", args.profile)

        for exp_name, (precision, engine_path) in trt_models.items():
            run_benchmark(
                TRTModel(engine_path, DEVICE), dataloader, DEVICE,
                f"{exp_name}_bs{bs}", precision, args.profile
            )

    summary_path = RESULTS_DIR / "summary.json"
    results = [BenchmarkResult(**d) for d in json.loads(summary_path.read_text())]
    print_results(results)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
