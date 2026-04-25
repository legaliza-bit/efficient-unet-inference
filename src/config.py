import torch
from pathlib import Path

COMPETITION = "carvana-image-masking-challenge"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
DATA_DIR = PROJECT_ROOT / "data"
IMGS_DIR = DATA_DIR / "train"
MASKS_DIR = DATA_DIR / "train_masks"

CKPT_PATH = TMP_DIR / "unet_carvana.pt"
QAT_CKPT_PATH = TMP_DIR / "unet_carvana_qat.pt"
ONNX_PATH = TMP_DIR / "unet_carvana.onnx"
TRT_FP16_PATH = TMP_DIR / "unet_carvana_fp16.engine"
TRT_FP8_PATH = TMP_DIR / "unet_carvana_fp8.engine"
TRT_INT8_PATH = TMP_DIR / "unet_carvana_int8.engine"
CALIB_PATH = TMP_DIR / "calib_data.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROFILE_DIR = TMP_DIR / "profiles"
RESULTS_DIR = TMP_DIR / "results"
IMG_SCALE = 0.5
BATCH_SIZE = 8
NUM_CLASSES = 2

FINETUNE_EPOCHS = 30
QAT_EPOCHS = 5
LR = 1e-4
