import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config import DEVICE, BATCH_SIZE, DATA_DIR, ONNX_PATH, ORT_FP16_PATH, ORT_INT8_PATH
from src.utils import load_model, print_results, download_carvana, reset_gpu_state
from src.data import get_carvana
from src.finetune.finetune import finetune, finetune_qat
from src.model import (
    apply_compiled, apply_ptq,
    export_to_onnx, apply_ort_fp16, apply_ort_int8, ORTModel,
)
from src.run_benchmark import run_benchmark

_CALIB_SAMPLES = 200
_CPU = torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune-qat", action="store_true")
    parser.add_argument("--download", action="store_true",
                        help="Download Carvana from Kaggle (requires ~/.kaggle/kaggle.json)")
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
    if DEVICE.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

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
    val_loader_cpu = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    calib_loader = DataLoader(
        Subset(val_ds, range(min(_CALIB_SAMPLES, len(val_ds)))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    sample, _ = next(iter(calib_loader))
    export_to_onnx(model, sample, ONNX_PATH)

    results = []

    def bench(m, loader, device, name, precision, **kw):
        print(f"\n── {name} {'─' * max(0, 52 - len(name))}")
        r = run_benchmark(m, loader, device, name, precision,
                          profile=args.profile, **kw)
        results.append(r)
        del m
        torch._dynamo.reset()
        reset_gpu_state()

    # ── 1. FP16 baseline (GPU) ────────────────────────────────────
    bench(model, val_loader, DEVICE, "fp16_baseline", "fp16")

    # ── 2. BF16 baseline (GPU) ───────────────────────────────────
    bench(model, val_loader, DEVICE, "bf16_baseline", "bf16")

    # ── 3. torch.compile FP16 (GPU) ──────────────────────────────
    print("\nCompiling model (first run will be slow)…")
    bench(apply_compiled(model), val_loader, DEVICE, "compile_fp16", "fp16")

    # ── 4. ORT FP16 (GPU) ────────────────────────────────────────
    apply_ort_fp16(ONNX_PATH, ORT_FP16_PATH)
    bench(ORTModel(ORT_FP16_PATH, DEVICE), val_loader, DEVICE, "ort_fp16", "fp16")

    # ── 5. ORT INT8 (GPU) ────────────────────────────────────────
    apply_ort_int8(ONNX_PATH, ORT_INT8_PATH, calib_loader)
    bench(ORTModel(ORT_INT8_PATH, DEVICE), val_loader, DEVICE, "ort_int8", "int8")

    # ── 6. PyTorch dynamic INT8 (CPU) ────────────────────────────
    bench(apply_ptq(model), val_loader_cpu, _CPU, "ptq_int8_cpu", "int8")

    print_results(results)

    if args.profile:
        from src.config import PROFILE_DIR
        print(f"\nChrome traces saved to {PROFILE_DIR}/")
        print("Open at chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
