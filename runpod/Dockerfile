# LTX-2 RunPod Serverless Dockerfile
# Optimized for A100/H100 (80GB) or A10G/RTX4090 (24GB) with FP8

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/runpod-volume/huggingface \
    MODEL_DIR=/runpod-volume/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the repository
COPY . /app/

# Install dependencies using uv
RUN uv sync --frozen

# Install additional RunPod dependencies
RUN uv pip install runpod requests

# Create directories for models
RUN mkdir -p /runpod-volume/models /runpod-volume/huggingface /runpod-volume/outputs

# Copy handler and download scripts
COPY runpod/handler.py /app/handler.py
COPY runpod/download_models.py /app/download_models.py

# Expose port for health checks
EXPOSE 8000

# Set the entrypoint
CMD ["python", "-u", "handler.py"]
