import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

from src.config import IMG_SIZE, NUM_EVAL_SAMPLES


class VOCSegDataset(Dataset):
    def __init__(self, root: str, image_set: str = "val", img_size: int = IMG_SIZE):
        self.ds = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=False,
        )
        self.img_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_size = img_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img = self.img_tf(img)

        mask = np.array(mask)
        mask = Image.fromarray(mask)
        mask = mask.resize((self.mask_size, self.mask_size), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask).astype(np.int64))

        return img, mask


def get_fixed_eval_dataset(
    root: str = "data",
    image_set: str = "val",
    num_samples: int = NUM_EVAL_SAMPLES,
) -> Dataset:
    full_ds = VOCSegDataset(root=root, image_set=image_set)
    n = min(num_samples, len(full_ds))
    return Subset(full_ds, list(range(n)))
