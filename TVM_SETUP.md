# TVM (Apache TVM) Integration

## Overview

This project includes Apache TVM as an alternative compiler backend for UNet inference optimization (point 3 of the project requirements). Since TVM requires Python ≤3.11 while the main project uses Python 3.13, the TVM benchmark runs as a subprocess under a separate Python 3.11 virtual environment (`.venv-tvm311`).

All TVM results are evaluated on the exact same `tmp/bench_cache.pt` (the pre-processed dataset cache in RAM) as the PyTorch tests to ensure fair and purely compute-bound metrics.

## System Prerequisites

Before building TVM, ensure your system has the following installed:

- cmake >= 3.10
- make
- g++ (C++17 support)
- CUDA Toolkit (compatible with your NVIDIA driver)
- cuDNN development headers
- cuBLAS development headers
- ~10 GB free disk space (for TVM build + LLVM download)
- Python 3.11 (`uv python install 3.11`)

## Building TVM from Source (Reproducibility)

TVM must be built from source with CUDA, cuDNN, and cuBLAS support to prevent falling back to slow generic kernels. Pre-built PyPI wheels are CPU-only.

```bash
# 1. Create Python 3.11 venv specifically for TVM
uv python install 3.11
uv venv .venv-tvm311 --python 3.11

# 2. Install dependencies
uv pip install --python .venv-tvm311/bin/python -r requirements-tvm.txt

# 3. Download LLVM 17.0.6 backend (Ubuntu 22.04 x86_64 example)
mkdir -p tmp && cd tmp
python3 -c "
import urllib.request
url = 'https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz'
urllib.request.urlretrieve(url, 'llvm.tar.xz')
"
tar xf llvm.tar.xz

# 4. Clone and build TVM v0.12.0
git clone --recursive https://github.com/apache/tvm.git tvm-src --depth 1 --branch v0.12.0
cd tvm-src && mkdir -p build && cp cmake/config.cmake build/
cd build

# Edit config.cmake for GPU support
sed -i 's/USE_CUDA=OFF/USE_CUDA=ON/' config.cmake
sed -i 's/USE_CUDNN=OFF/USE_CUDNN=ON/' config.cmake
sed -i 's/USE_CUBLAS=OFF/USE_CUBLAS=ON/' config.cmake
sed -i "s|USE_LLVM=OFF|USE_LLVM=$(pwd)/../../clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/bin/llvm-config|" config.cmake

# (Optional) If system only provides versioned linker names:
mkdir -p ../../local-lib
ln -sf /usr/lib/x86_64-linux-gnu/libzstd.so.1 ../../local-lib/libzstd.so
ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 ../../local-lib/libtinfo.so
ln -sf /usr/lib/x86_64-linux-gnu/libxml2.so.2 ../../local-lib/libxml2.so

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_SHARED_LINKER_FLAGS="-L$(pwd)/../../local-lib" .. && make -j$(nproc)

# Return to project root
cd ../../..
```

## Usage

### 1. Run full benchmark suite with TVM (no tuning)
```bash
uv run python -m src.main --tvm
```

### 2. Run TVM benchmarks with AutoTVM tuning (slow compilation, fast execution)
AutoTVM tuning iterates over XGBoost models to find the fastest CUDA kernel configurations. These are cached in `tmp/tvm_tuning_logs/` and `tmp/tvm_cache/`.

```bash
uv run python -m src.main --tvm --tvm-tune --tvm-tune-trials 1000
```

### 3. Direct TVM debugging
If you need to bypass the main pipeline and debug TVM directly:

```bash
export PYTHONPATH=tmp/tvm-src/python:tmp/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib/python3.11/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=tmp/tvm-src/build:tmp/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib:tmp/local-lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

.venv-tvm311/bin/python src/_tvm_benchmark.py \
    --onnx-path tmp/unet_carvana.onnx \
    --precision fp16 \
    --output-dir tmp/tvm_results \
    --batch-size 8 \
    --cache-path tmp/bench_cache.pt \
    --num-workers 2 \
    --max-batches 0
```

## Compilation Pipeline

1. **ONNX Import**: `tvm.relay.frontend.from_onnx()` converts ONNX → Relay IR.
2. **FP16 Conversion**: `relay.transform.ToMixedPrecision("float16", missing_op_mode=1)`. Loss-sensitive ops remain in FP32.
3. **Optimization**: SimplifyInference → FoldConstant → FuseOps → CombineParallelConv2D.
4. **CUDA Target Selection**: auto-detect `sm_XX` and enable `-libs=cudnn,cublas` when available.
5. **Artifact Cache**: save/load compiled graph executor artifacts from `tmp/tvm_cache/`.
6. **AutoTVM Tuning**: per-model cached logs with resume support (`--tvm-tune`).
7. **CUDA Compilation**: `relay.build(target="cuda -arch=sm_XX -libs=cudnn,cublas", opt_level=4)`.