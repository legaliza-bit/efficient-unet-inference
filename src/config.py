import torch 
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 16
FINETUNE_EPOCHS = 2
LR = 1e-4
WARMUP_ITERS = 20
DATASET_CACHE = {}
