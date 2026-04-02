FROM registry.mipmap3d.com/docker/mipmap/runtime:v5.0.3.2-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch-cache \
    HF_HOME=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace/VGGT-X

COPY requirements.txt pyproject.toml README.md ./
COPY utils ./utils
COPY vggt ./vggt
COPY demo_colmap.py ./
COPY colmap_parallel.sh ./
COPY colmap_viser.py ./

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

RUN python3 -m pip install \
    numpy==1.26.1 \
    viser==0.2.23 \
    Pillow \
    huggingface_hub \
    einops \
    safetensors \
    opencv-python \
    open3d \
    evo \
    roma \
    kornia \
    trimesh \
    matplotlib \
    pycolmap==3.10.0 \
    pyliblzfse \
    tqdm \
    hydra-core \
    omegaconf \
    scipy \
    onnxruntime \
    requests

RUN python3 -m pip install -e .

CMD ["python3", "demo_colmap.py", "--help"]
