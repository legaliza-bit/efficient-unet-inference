from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
DATA_DIR = PROJECT_ROOT / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
NUM_CLASSES = 21
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
WARMUP_ITERS = 20
NUM_EVAL_SAMPLES = 200
