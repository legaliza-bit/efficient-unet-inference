import argparse

import torch
from torch.utils.data import DataLoader, Subset

from src.config import DEVICE, BATCH_SIZE, DATA_DIR
from src.utils import load_model, print_results, download_carvana
from src.data import get_carvana
from src.finetune.finetune import finetune, finetune_qat
from src.model import build_model, apply_trt_int8
from src.run_benchmark import run_benchmark

_CALIB_SAMPLES = 200  # images used for TRT INT8 calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune", action="store_true",
        help="Finetune on Carvana and save checkpoint before experiments",
    )
    parser.add_argument(
        "--finetune-qat", action="store_true",
        help="Run QAT finetuning and save checkpoint before experiments",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download Carvana from Kaggle (requires ~/.kaggle/kaggle.json)",
    )
    args = parser.parse_args()

    carvana_dir = DATA_DIR / "carvana"
    imgs_dir = carvana_dir / "imgs"

    if imgs_dir.exists():
        print(f"Found existing dataset at {carvana_dir}")
    elif args.download:
        download_carvana(carvana_dir)
    else:
        raise FileNotFoundError(
            f"No dataset at {carvana_dir}. Run with --download to fetch it from Kaggle."
        )

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.finetune:
        print("\n── Finetuning ──────────────────────────────────")
        finetune(build_model())

    if args.finetune_qat:
        print("\n── QAT Finetuning ───────────────────────────────")
        finetune_qat(load_model())

    val_ds = get_carvana("val")
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"),
    )
    calib_loader = DataLoader(
        Subset(val_ds, range(min(_CALIB_SAMPLES, len(val_ds)))),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(DEVICE.type == "cuda"),
    )

    model = load_model()

    print("\n── Experiment 1: FP16 baseline (GPU) ───────────────────")
    results_fp16 = run_benchmark(
        model, val_loader, DEVICE,
        pipeline_name="fp16_baseline",
        use_fp16=True,
    )

    print("\n── Experiment 2: TensorRT INT8 (GPU) ───────────────────")
    trt_model = apply_trt_int8(model, calib_loader, DEVICE)
    results_trt = run_benchmark(
        trt_model, val_loader, DEVICE,
        pipeline_name="trt_int8",
    )

    print_results([results_fp16, results_trt])


if __name__ == "__main__":
    main()
