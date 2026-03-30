from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from src.config import DATASET_CACHE


def get_voc_dataset(**kwargs):
    key = str(kwargs)

    if key not in DATASET_CACHE:
        DATASET_CACHE[key] = VOCMultiClassDataset(**kwargs)
        print("Loaded dataset")
    else:
        print("Using cached dataset")

    return DATASET_CACHE[key]


class VOCMultiClassDataset(Dataset):
    def __init__(self, root, image_set="train", img_size=256):
        self.ds = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=True
        )

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.mask_size = img_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]

        img = self.img_tf(img)

        mask = np.array(mask)

        mask = Image.fromarray(mask)
        mask = mask.resize(
            (self.mask_size, self.mask_size),
            resample=Image.NEAREST
        )

        mask = np.array(mask).astype(np.int64)

        mask = torch.from_numpy(mask)

        return img, mask
