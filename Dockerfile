# Audio Forensic: web app + CLI. GPU: run with the NVIDIA Container Toolkit (see README).
FROM python:3.12-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/root/.cache/huggingface \
    AF_HOST=0.0.0.0 \
    AF_PORT=8000 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ffmpeg: decoding/encoding · git: git-hosted Python deps · nodejs: yt-dlp's YouTube JS challenges
# libvulkan1 + vulkan-tools: audio.cpp (GGUF models) on the GPU
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git nodejs libvulkan1 vulkan-tools ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/
WORKDIR /app

# Dependencies first (cached layer), then the project. The uv cache is a build cache, not part of the image.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

EXPOSE 8000
VOLUME ["/app/data", "/app/models", "/app/vendor", "/root/.cache/huggingface"]
CMD ["uv", "run", "--no-sync", "audio-forensic-web"]
