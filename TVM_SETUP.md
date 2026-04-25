# TVM (Apache TVM) Integration

## Overview

This project includes Apache TVM as an alternative compiler backend for UNet inference optimization (point 3 of the project requirements). Since TVM requires Python ≤3.11 while the main project uses Python 3.13, the TVM benchmark runs as a subprocess under a separate Python 3.11 virtual environment.

## Architecture

```
main.py (--tvm flag, Python 3.13)
  └── src/tvm.py (subprocess launcher)
      └── .venv-tvm311/bin/python src/_tvm_benchmark.py (Python 3.11)
          ├── ONNX → TVM Relay import
          ├── FP16 mixed-precision conversion (ToMixedPrecision)
          ├── Optional AutoTVM tuning (--tune flag)
          ├── TVM CUDA compilation (LLVM codegen, auto-detected sm_XX)
          └── Dual-timing benchmark + quality metrics → JSON results
```

## Changes from v1 (Performance Fixes)

| Issue | Before | After |
|-------|--------|-------|
| Dataset repeat | 10× (`ConcatDataset([val_ds] * 10)`) → ~625 batches | 1× → ~50 batches |
| `max_batches` default | 0 (run ALL batches, ~43 min/precision) | 10 (run 10 batches, ~42 sec/precision) |
| Timing methodology | Included H2D + D2H transfers | Dual: compute-only (like PyTorch) + end-to-end |
| TVM CUDA backend | Generic CUDA kernels only | `-libs=cudnn,cublas` vendor kernels on CUDA |
| Build reuse | Recompiled every run | Compiled graph executor artifacts cached in `tmp/tvm_cache/` |
| Dataset parity | TVM loader diverged from `src/data.py` | Same image/mask loading logic as PyTorch pipeline |
| AutoTVM tuning | Partial/fragile | Per-model cached logs, resume support, includes `conv2d_transpose`/Winograd tasks |
| `--tvm-tune` flag | N/A | Added to `main.py` CLI |

## Benchmark Results

### Comparison with PyTorch FP16 Baseline

| Metric | PyTorch FP16 | TVM FP16 (compute) | TVM FP16 (e2e) | TVM FP32 (compute) | TVM FP32 (e2e) |
|--------|-------------|-------------------|----------------|-------------------|----------------|
| Latency mean (ms) | 24.53 | 36.43 | 41.82 | 57.71 | 64.58 |
| Throughput (samples/s) | 649.0 | 219.6 | 191.3 | 138.6 | 123.9 |
| mIoU | ~0.97-0.98 | 0.9865 | 0.9865 | 0.9865 | 0.9865 |
| Dice | ~0.98-0.99 | 0.9932 | 0.9932 | 0.9932 | 0.9932 |
| Compile time (s) | — | 16.0 first run / 3.3 cached | — | 17.3 | — |
| Tuned | — | No | No | No | No |

> **Note**: "compute" timing only measures `module.run()` (comparable to PyTorch CUDA events). "e2e" includes H2D set_input + run + D2H get_output.
>
> These TVM numbers were measured with `--max-batches 1` for fast iteration while debugging the backend.

### Key Findings

1. **The original 4-second latency was caused mainly by backend selection, not by model quality.**
   TVM had been built/run without `cuDNN/cuBLAS`, so it fell back to slow generic CUDA kernels. Rebuilding TVM locally with vendor libraries and compiling with `-libs=cudnn,cublas` reduced FP16 compute latency from ~4149 ms to ~36 ms.

2. **The low mIoU/Dice was caused by a dataset bug in the TVM subprocess, not by ONNX/Relay numerical drift.**
   The TVM loader handled Carvana palette masks incorrectly and collapsed them to background-only masks. After matching the mask loading logic from `src/data.py`, metrics returned to the expected range (`mIoU=0.9865`, `Dice=0.9932`).

3. **Data transfer overhead remains modest** for this setup: FP16 compute latency (36.43 ms) vs e2e (41.82 ms) differs by ~5.4 ms, so H2D/D2H is measurable but not dominant.

4. **FP16 is faster than FP32 in the fixed TVM path**: `36.43 ms` vs `57.71 ms` compute latency in the short check, with identical quality on the tested batch.

5. **Iteration is now practical**: the compiled artifact cache avoids rebuilding the same model on every short check, reducing a repeated FP16 "compile" stage from ~16.0 s to ~3.3 s.

## Prerequisites

### 1. TVM Built from Source

TVM must be built from source with CUDA support (pre-built PyPI wheels are CPU-only).

**Build location:** `tmp/tvm-src/`

**Build configuration:**
- TVM v0.12.0
- CUDA support (`USE_CUDA=ON`)
- cuDNN support (`USE_CUDNN=ON`)
- cuBLAS support (`USE_CUBLAS=ON`)
- LLVM 17.0.6 backend (`USE_LLVM=<path>`)
- Graph executor (`USE_GRAPH_EXECUTOR=ON`)

### 2. Python 3.11 Virtual Environment

Location: `.venv-tvm311/`

Required packages:
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `numpy<2` (1.26.4, TVM compatibility)
- `onnx` (1.16.2)
- `pillow`, `loguru`

### 3. Building TVM from Source (Reproducibility)

```bash
# Install Python 3.11
uv python install 3.11

# Create venv
uv venv .venv-tvm311 --python 3.11

# Install dependencies
.venv-tvm311/bin/pip install "numpy<2" onnx onnxoptimizer pillow loguru
.venv-tvm311/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Download LLVM 17.0.6
mkdir -p tmp && cd tmp
python3 -c "
import urllib.request
url = 'https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz'
urllib.request.urlretrieve(url, 'llvm.tar.xz')
"
tar xf llvm.tar.xz

# Clone and build TVM
git clone --recursive https://github.com/apache/tvm.git tvm-src --depth 1 --branch v0.12.0
cd tvm-src && mkdir -p build && cp cmake/config.cmake build/
cd build

# Edit config.cmake
sed -i 's/USE_CUDA=OFF/USE_CUDA=ON/' config.cmake
sed -i 's/USE_CUDNN=OFF/USE_CUDNN=ON/' config.cmake
sed -i 's/USE_CUBLAS=OFF/USE_CUBLAS=ON/' config.cmake
sed -i "s|USE_LLVM=OFF|USE_LLVM=$(pwd)/../../clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/bin/llvm-config|" config.cmake

# If the system only provides versioned linker names, create local symlinks once
mkdir -p ../../local-lib
ln -sf /usr/lib/x86_64-linux-gnu/libzstd.so.1 ../../local-lib/libzstd.so
ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 ../../local-lib/libtinfo.so
ln -sf /usr/lib/x86_64-linux-gnu/libxml2.so.2 ../../local-lib/libxml2.so

cmake -DCMAKE_SHARED_LINKER_FLAGS="-L$(pwd)/../../local-lib" .. && make -j$(nproc)
```

## Usage

### Run TVM benchmarks (default: 10 batches, no tuning)
```bash
uv run python -m src.main --tvm
```

### Run TVM benchmarks with AutoTVM tuning (slow but may improve performance)
```bash
uv run python -m src.main --tvm --tvm-tune
```

### Direct TVM benchmark (debugging)
```bash
export PYTHONPATH=tmp/tvm-src/python:$PYTHONPATH
export LD_LIBRARY_PATH=tmp/tvm-src/build:tmp/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib:tmp/local-lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

.venv-tvm311/bin/python src/_tvm_benchmark.py \
    --onnx-path tmp/unet_carvana.onnx \
    --precision fp16 \
    --output-dir tmp/tvm_results \
    --batch-size 8 \
    --data-dir data/carvana \
    --img-scale 0.5 \
    --num-workers 2 \
    --max-batches 10

# With AutoTVM tuning
.venv-tvm311/bin/python src/_tvm_benchmark.py \
    --onnx-path tmp/unet_carvana.onnx \
    --precision fp16 \
    --output-dir tmp/tvm_results \
    --batch-size 8 \
    --data-dir data/carvana \
    --img-scale 0.5 \
    --num-workers 2 \
    --max-batches 10 \
    --tune
```

## Compilation Pipeline

1. **ONNX Import**: `tvm.relay.frontend.from_onnx()` converts ONNX → Relay IR
2. **FP16 Conversion** (optional): `relay.transform.ToMixedPrecision("float16", missing_op_mode=1)`
3. **Optimization**: SimplifyInference → FoldConstant → FuseOps → CombineParallelConv2D
4. **CUDA Target Selection**: auto-detect `sm_XX` and enable `-libs=cudnn,cublas` when available
5. **Artifact Cache**: save/load compiled graph executor artifacts from `tmp/tvm_cache/`
6. **AutoTVM Tuning** (optional, `--tune`): per-model cached logs with resume support
7. **CUDA Compilation**: `relay.build(target="cuda -arch=sm_XX -libs=cudnn,cublas", opt_level=4)`

## Timing Methodology

The benchmark reports two timing measurements:

| Metric | What's measured | Comparable to |
|--------|----------------|---------------|
| `latency_compute_*` | Only `module.run()` (GPU computation) | PyTorch CUDA Events timing |
| `latency_e2e_*` | `set_input()` + `module.run()` + `get_output()` | Real production latency |

For this UNet model, data transfer overhead is minimal (~5 ms / ~0.1% of total), so both metrics are similar.

## Files

| File | Description |
|------|-------------|
| `src/tvm.py` | Integration module — launches TVM subprocess |
| `src/_tvm_benchmark.py` | Standalone benchmark (Python 3.11) |
| `src/config.py` | Added `TVM_RESULTS_DIR` |
| `src/main.py` | Added `--tvm` and `--tvm-tune` flags |
| `TVM_SETUP.md` | This documentation |

## Compatibility Notes

- **TVM 0.12.0**: `relay.transform.Sequential` moved to `tvm.transform.Sequential` (compat shim included)
- **CUDA Driver**: ctypes init before TVM import
- **numpy**: Requires < 2 (incompatible with numpy 2.x)
- **Import shadowing**: `src/tvm.py` shadows the `tvm` package; the benchmark script removes `src/` from `sys.path`