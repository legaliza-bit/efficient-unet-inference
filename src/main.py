import argparse

import torch
from torch.utils.data import DataLoader, Subset

from src.config import DEVICE, BATCH_SIZE, DATA_DIR, ONNX_PATH, ORT_INT8_PATH
from src.utils import load_model, print_results, download_carvana
from src.data import get_carvana
from src.finetune.finetune import finetune, finetune_qat
from src.model import build_model, export_to_onnx, apply_ort_int8, ORTModel
from src.run_benchmark import run_benchmark

_CALIB_SAMPLES = 200


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
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    model = load_model()

    print("\n── Experiment 1: FP16 baseline (GPU) ───────────────────")
    results_fp16 = run_benchmark(
        model, val_loader, DEVICE,
        pipeline_name="fp16_baseline",
        use_fp16=True,
    )

    print("\n── Experiment 2: ORT INT8 (GPU) ─────────────────────────")
    sample, _ = next(iter(calib_loader))
    export_to_onnx(model, sample, ONNX_PATH)
    apply_ort_int8(ONNX_PATH, ORT_INT8_PATH, calib_loader)
    ort_model = ORTModel(ORT_INT8_PATH, DEVICE)
    results_ort = run_benchmark(
        ort_model, val_loader, DEVICE,
        pipeline_name="ort_int8",
    )

    print_results([results_fp16, results_ort])


if __name__ == "__main__":
    main()
