#!/bin/bash
set -e
uv python install 3.11
uv venv .venv-tvm311 --python 3.11
uv pip install --python .venv-tvm311/bin/python -r requirements-tvm.txt

mkdir -p tmp
cd tmp

if [ ! -f "llvm.tar.xz" ]; then
    echo "Downloading LLVM..."
    python3 -c "
import urllib.request
url = 'https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz'
urllib.request.urlretrieve(url, 'llvm.tar.xz')
"
    tar xf llvm.tar.xz
fi

if [ ! -d "tvm-src" ]; then
    git clone --recursive https://github.com/apache/tvm.git tvm-src --depth 1 --branch v0.12.0
fi

cd tvm-src
mkdir -p build
cp cmake/config.cmake build/
cd build

sed -i 's/USE_CUDA=OFF/USE_CUDA=ON/' config.cmake
sed -i 's/USE_CUDNN=OFF/USE_CUDNN=ON/' config.cmake
sed -i 's/USE_CUBLAS=OFF/USE_CUBLAS=ON/' config.cmake
sed -i "s|USE_LLVM=OFF|USE_LLVM=$(pwd)/../../clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/bin/llvm-config|" config.cmake

mkdir -p ../../local-lib
ln -sf /usr/lib/x86_64-linux-gnu/libzstd.so.1 ../../local-lib/libzstd.so
ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 ../../local-lib/libtinfo.so
ln -sf /usr/lib/x86_64-linux-gnu/libxml2.so.2 ../../local-lib/libxml2.so

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_SHARED_LINKER_FLAGS="-L$(pwd)/../../local-lib" ..
make -j$(nproc)

echo "TVM build complete!"
