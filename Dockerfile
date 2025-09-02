FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-setuptools python3-dev \
        git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.6 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

CMD ["python3"]