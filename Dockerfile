FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv python install 3.13

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

COPY . .

CMD ["uv", "run", "python", "-m", "src.main"]
