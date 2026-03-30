FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /src

RUN apt-get update && apt-get install -y \
    python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# 🔥 PyTorch CUDA explicitly
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .

COPY . .

CMD ["python3", "main.py"]