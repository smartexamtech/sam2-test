FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/sam2.git

# Download SAM 2 checkpoint
RUN mkdir -p /app/checkpoints && \
    wget -P /app/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Copy the script and videos
COPY extract.py .
COPY videos /app/videos

# Default command
CMD ["python", "extract.py"]