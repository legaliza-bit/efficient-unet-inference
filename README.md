# Efficient UNet Inference

Benchmarking inference optimization techniques for UNet on the Carvana segmentation dataset.

**Team:** Artur Gimranov, Elizaveta Romanko, Niyaz Bureev

---

## Setup

```bash
uv sync
```

Requires CUDA-capable GPU. Tested on PyTorch 2.11 + CUDA 12.8.

---

## Running the benchmark

The benchmark loads a preprocessed cache of 500 Carvana samples (downloaded automatically from Google Drive).

```bash
uv run python -m src.main
```

Optional flags:

| Flag | Description |
|------|-------------|
| `--batch-sizes 1 4 8 16` | Sweep over batch sizes (default: 8) |
| `--trt` | Include TensorRT experiments (requires `uv sync --extra trt`) |
| `--profile` | Save Chrome traces to `tmp/profiles/` |
| `--finetune` | Fine-tune before benchmarking (requires raw Carvana data) |

### Preparing the cache from scratch (optional)

Requires Carvana data from Kaggle (see Kaggle API setup below).

```bash
uv run python -m src.main --prepare-cache --download
```

### Batch size study

```bash
uv run python -m src.batch_size_study
uv run python -m src.batch_size_study --batch-sizes 1 2 4 8 16 32 64
```

Saves a 4-panel plot to `tmp/results/batch_size_study.png`.

---

## Results

Selected experiments at batch size 8 on NVIDIA A100:

| Experiment | Prec | Lat p50 (ms) | Lat p95 (ms) | Lat p99 (ms) | Tput (img/s) | GPU mem (MB) | mIoU | Dice |
|---|---|---|---|---|---|---|---|---|
| baseline_fp32_bs8 | fp32 | 63.98 | 64.06 | 64.10 | 125.0 | 11687 | 0.9287 | 0.9626 |
| compile_fp16_bs8 | fp16 | 17.98 | 18.43 | 802.71 | 156.2 | 2756 | 0.9288 | 0.9626 |
| pruning_70pct_bs8 | fp32 | 63.83 | 63.94 | 63.97 | 125.3 | 11805 | 0.9196 | 0.9575 |
| tvm_fp16_bs8 | fp16 | 36.28 | 36.32 | 38.98 | 220.5 | 7204 | 0.9285 | 0.9625 |
| trt_int8_bs8 | int8 | 8.16 | 8.26 | 8.30 | 978.3 | 570 | 0.9229 | 0.9594 |

For a full analysis and methodology, see [presentation.pdf](presentation.pdf).

---

## Kaggle API setup

Required only for `--download` / `--prepare-cache`.

1. Join the [Carvana competition](https://www.kaggle.com/competitions/carvana-image-masking-challenge) and accept the rules.
2. Go to Kaggle → Settings → API → **Create New Token**.
3. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.

