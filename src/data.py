from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

from src.config import DATA_DIR, IMG_SCALE


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
    imgs_dir = DATA_DIR / "carvana" / "imgs"
    masks_dir = DATA_DIR / "carvana" / "masks"

    ds = CarvanaDataset(imgs_dir, masks_dir, scale=scale)
    n_val = max(1, int(len(ds) * val_fraction))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    return train_ds if split == "train" else val_ds
