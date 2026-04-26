from pathlib import Path
import json
import zipfile
import subprocess
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.transforms import v2
from torchvision import tv_tensors
from loguru import logger

from src.config import DATA_DIR, IMG_SCALE, COMPETITION, IMGS_DIR, MASKS_DIR, CACHE_PATH, GDRIVE_FILE_ID, BENCH_N_SAMPLES


def _train_augmentations() -> v2.Compose:
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.RandomPerspective(distortion_scale=0.2, p=0.3),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    ])


class AugmentedSubset(Dataset):
    def __init__(self, subset, transform: v2.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        img, mask = self.transform(tv_tensors.Image(img), tv_tensors.Mask(mask))
        return img.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor)


class CarvanaDataset(Dataset):
    """
    Carvana binary segmentation dataset.
    Expects:
      data/carvana/imgs/   — JPEG images
      data/carvana/masks/  — GIF masks named <id>_mask.gif (values 0 and 255)

    Mask class mapping: 0 → background, 1 → car.
    Images are normalised to [0, 1] and scaled by IMG_SCALE.
    """

    MASK_VALUES = [0, 255]

    def __init__(self, images_dir: Path, masks_dir: Path, scale: float = IMG_SCALE):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.scale = scale

        self.ids = sorted(
            f.stem for f in images_dir.iterdir() if not f.name.startswith(".")
        )
        if not self.ids:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _preprocess(pil_img: Image.Image, scale: float, is_mask: bool) -> np.ndarray:
        w, h = pil_img.size
        newW, newH = max(1, int(scale * w)), max(1, int(scale * h))
        resample = Image.NEAREST if is_mask else Image.BICUBIC
        pil_img = pil_img.resize((newW, newH), resample=resample)
        arr = np.asarray(pil_img)

        if is_mask:
            out = np.zeros((newH, newW), dtype=np.int64)
            for idx, v in enumerate(CarvanaDataset.MASK_VALUES):
                out[arr == v] = idx
            return out
        else:
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            else:
                arr = arr.transpose((2, 0, 1))
            if arr.max() > 1:
                arr = arr / 255.0
            return arr.astype(np.float32)

    def __getitem__(self, idx: int):
        name = self.ids[idx]

        img_files = list(self.images_dir.glob(name + ".*"))
        mask_files = list(self.masks_dir.glob(name + "_mask.*"))
        if not img_files:
            raise FileNotFoundError(f"No image found for id '{name}' in {self.images_dir}")
        if not mask_files:
            raise FileNotFoundError(f"No mask found for id '{name}' in {self.masks_dir}")

        img = Image.open(img_files[0]).convert("RGB")
        mask = Image.open(mask_files[0]).convert("L")

        img_arr = self._preprocess(img, self.scale, is_mask=False)
        mask_arr = self._preprocess(mask, self.scale, is_mask=True)

        return (
            torch.as_tensor(img_arr.copy()).float().contiguous(),
            torch.as_tensor(mask_arr.copy()).long().contiguous(),
        )


def get_carvana(split: str = "train", val_fraction: float = 0.1, scale: float = IMG_SCALE):
    ds = CarvanaDataset(IMGS_DIR, MASKS_DIR, scale=scale)
    n_val = max(1, int(len(ds) * val_fraction))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    if split == "train":
        return AugmentedSubset(train_ds, _train_augmentations())
    return val_ds


class CachedDataset(Dataset):
    def __init__(self, cache_path: Path = CACHE_PATH):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.imgs = data["imgs"]    # float16
        self.masks = data["masks"]  # uint8

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx].float(), self.masks[idx].long()


def prepare_benchmark_cache(
    cache_path: Path = CACHE_PATH,
    n_samples: int = BENCH_N_SAMPLES,
) -> None:
    """Preprocess n_samples from Carvana once, save to cache_path for reproducible benchmarking."""
    ds = CarvanaDataset(IMGS_DIR, MASKS_DIR, scale=IMG_SCALE)
    indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))[:n_samples].tolist()

    imgs, masks = [], []
    for i, idx in enumerate(indices):
        img, mask = ds[idx]
        imgs.append(img)
        masks.append(mask)
        if (i + 1) % 50 == 0:
            logger.info(f"Prepared {i + 1}/{n_samples} samples...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"imgs": torch.stack(imgs).half(), "masks": torch.stack(masks).to(torch.uint8)},
        cache_path,
    )
    size_mb = cache_path.stat().st_size / 1024**2
    logger.info(f"Saved {n_samples} samples ({size_mb:.0f} MB) → {cache_path}")


def download_bench_cache(cache_path: Path = CACHE_PATH) -> None:
    """Download benchmark cache from Google Drive (set GDRIVE_FILE_ID in config.py)."""
    if not GDRIVE_FILE_ID:
        raise ValueError("GDRIVE_FILE_ID is not set in config.py")
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required: uv sync")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading benchmark cache from Google Drive...")
    gdown.download(id=GDRIVE_FILE_ID, output=str(cache_path), quiet=False)


def download_carvana() -> None:
    """Download Carvana images and masks from Kaggle into dest_dir."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in "train", "train_masks":
        zip_path = DATA_DIR / f"{filename}.zip"

        if not zip_path.exists():
            logger.info(f"Downloading {filename} …")
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", f"{filename}.zip", "-p", str(DATA_DIR)],
                check=True,
            )

        logger.info(f"Extracting {filename} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    logger.info("Carvana dataset ready.")
