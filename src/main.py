import torch

from src.run_benchmark import run_benchmark
from src.config import DEVICE, BATCH_SIZE
from src.data import get_voc_dataset
from src.finetune import finetune
from src.models.baseline import UNetResNet18
from torch.utils.data import DataLoader


def main():
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Memory: {total_mem:.1f} GB")

    print()

    print("── Loading dataset ──────────────────────────")

    test_ds = get_voc_dataset(image_set="val")

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = UNetResNet18()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: UNetResNet18 | {n_params / 1e6:.1f}M params")

    print("\n── Fine-tuning ─────────────────────────────")

    finetune(model=model, num_classes=21)

    print("\n── Benchmark ───────────────────────────────")

    results = run_benchmark(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        pipeline_name="unet_voc_multiclass_fp16",
        num_classes=21,
    )

    print("\n── Done ─────────────────────────────────────")

    return results


if __name__ == "__main__":
    main()
