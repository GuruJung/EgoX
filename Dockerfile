# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HUB_CACHE=/cache/huggingface/hub

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

RUN python -m pip install --upgrade pip

COPY requirements.txt ./
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 torchvision==0.25.0 \
    && python -m pip install -r requirements.txt \
    && python -m pip install \
        huggingface_hub \
        openai \
        tensorboard

COPY . .

CMD ["bash"]
