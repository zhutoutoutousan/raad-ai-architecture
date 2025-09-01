# Use PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for advanced models
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    timm==0.9.12 \
    kagglehub==0.1.0

# Copy source code
COPY src/ ./src/
COPY run_experiment.py .
COPY run_training.py .
COPY run_optimized_training.py .
COPY run_advanced_training.py .
COPY test_setup.py .

# Create necessary directories
RUN mkdir -p /app/data/processed /app/models/checkpoints /app/logs /app/checkpoints

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "run_advanced_training.py"]
