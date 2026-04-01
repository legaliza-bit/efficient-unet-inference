import torch 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
DATA_DIR = PROJECT_ROOT / "data"
CKPT_PATH = TMP_DIR / "unet_finetuned.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 16
FINETUNE_EPOCHS = 30
LR = 1e-4
WARMUP_ITERS = 20
DATASET_CACHE = {}
