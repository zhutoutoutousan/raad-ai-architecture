# 🚀 Architectural Research - Cloud Deployment Guide

This guide provides comprehensive instructions for deploying the Architectural Style Classification research environment to cloud servers with Docker Hub integration and hyperparameter optimization capabilities.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Docker Hub Setup](#docker-hub-setup)
- [Cloud Server Deployment](#cloud-server-deployment)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Monitoring and Scaling](#monitoring-and-scaling)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### 1. Local Setup (Docker Hub Preparation)

```bash
# Clone the repository
git clone <repository-url>
cd raad-ai-architecture

# Run deployment setup
chmod +x scripts/deploy_to_cloud.sh
./scripts/deploy_to_cloud.sh
```

### 2. Cloud Server Setup

```bash
# Upload to cloud server
scp -r . user@your-server:/path/to/research/

# SSH into server
ssh user@your-server

# Run cloud setup
cd /path/to/research
chmod +x cloud_setup.sh
./cloud_setup.sh
```

### 3. Deploy and Run

```bash
# Deploy services
./deploy.sh

# Start experiments
./scale_experiments.sh

# Monitor progress
./monitor.sh
```

## 🐳 Docker Hub Setup

### 1. Docker Hub Account

1. Create a Docker Hub account at [hub.docker.com](https://hub.docker.com)
2. Create a new repository: `architectural-research`
3. Get your Docker Hub credentials

### 2. Configure Credentials

Edit the deployment script:

```bash
# Edit scripts/deploy_to_cloud.sh
DOCKER_HUB_USERNAME="your-username"  # Change this
```

### 3. Build and Push

```bash
# Build production image
docker build -f Dockerfile.production -t architectural-research:latest .

# Tag for Docker Hub
docker tag architectural-research:latest your-username/architectural-research:latest

# Push to Docker Hub
docker push your-username/architectural-research:latest
```

### 4. Pull on Cloud Server

```bash
# Pull latest image
docker pull your-username/architectural-research:latest
```

## ☁️ Cloud Server Deployment

### 1. Server Requirements

**Minimum Requirements:**
- Ubuntu 20.04+ or CentOS 8+
- 8+ CPU cores
- 32GB+ RAM
- 100GB+ storage
- NVIDIA GPU (optional but recommended)
- Docker and Docker Compose

**Recommended:**
- 16+ CPU cores
- 64GB+ RAM
- 500GB+ SSD storage
- Multiple NVIDIA GPUs
- High-speed internet connection

### 2. Server Setup

#### Ubuntu/Debian

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Docker (if GPU available)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### CentOS/RHEL

```bash
# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. Environment Configuration

Create `.env` file:

```bash
# Docker Hub credentials
DOCKER_HUB_USERNAME=your-username

# Weights & Biases (optional)
WANDB_API_KEY=your-wandb-key

# Kaggle credentials (optional)
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key

# GPU configuration
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all available GPUs
```

### 4. Deploy Services

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

## 🔍 Hyperparameter Optimization

### 1. Optuna Optimization

```bash
# Run Optuna optimization
docker exec architectural-research-production python hyperparameter_optimization.py \
    --method optuna \
    --trials 100 \
    --epochs 20 \
    --data_path /app/data/real_architecture/processed
```

### 2. Ray Tune Optimization

```bash
# Run Ray Tune optimization
docker exec architectural-research-production python hyperparameter_optimization.py \
    --method ray_tune \
    --trials 50 \
    --epochs 15
```

### 3. Combined Optimization

```bash
# Run both optimization methods
docker exec architectural-research-production python hyperparameter_optimization.py \
    --method both \
    --trials 75 \
    --epochs 20
```

### 4. Monitor Optimization

```bash
# Access Ray Dashboard
# Open browser: http://your-server-ip:8265

# View optimization results
ls -la results/optuna_optimization_results.json
ls -la results/ray_tune_optimization_results.json
```

## 📊 Monitoring and Scaling

### 1. Real-time Monitoring

```bash
# Monitor system resources
./monitor.sh

# View GPU usage
nvidia-smi

# Check container status
docker ps

# View recent logs
docker logs --tail 50 architectural-research-production
```

### 2. TensorBoard Monitoring

```bash
# Access TensorBoard
# Open browser: http://your-server-ip:6006

# Or run TensorBoard locally
docker exec architectural-research-production tensorboard \
    --logdir=/app/logs \
    --host=0.0.0.0 \
    --port=6006
```

### 3. Scaling Experiments

```bash
# Run large-scale experiments
./scale_experiments.sh

# Or run individual experiments
docker exec architectural-research-production python scale_up_experiments.py \
    --experiment hierarchical_advanced \
    --epochs 100 \
    --batch_size 32 \
    --samples_per_class 1000
```

### 4. Multi-GPU Training

```bash
# Update docker-compose.production.yml
# Set CUDA_VISIBLE_DEVICES=0,1,2,3

# Restart services
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
```

## 🔧 Advanced Configuration

### 1. Custom Hyperparameter Ranges

Edit `hyperparameter_optimization.py`:

```python
# Modify search spaces
params = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
    'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128]),
    'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512, 1024]),
    # Add more parameters...
}
```

### 2. Custom Model Architectures

Edit `src/models/hierarchical_classifier.py`:

```python
# Add new architectural components
class CustomBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Custom implementation
        pass
```

### 3. Custom Loss Functions

Edit `src/training/losses.py`:

```python
class CustomLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Custom loss implementation
        pass
```

## 🚨 Troubleshooting

### 1. Common Issues

#### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Install nvidia-docker2 if needed
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

#### Memory Issues

```bash
# Reduce batch size
docker exec architectural-research-production python run_experiment.py \
    --batch_size 8

# Use gradient accumulation
# Edit config to use accumulate_grad_batches=4
```

#### Network Issues

```bash
# Check Docker network
docker network ls

# Restart Docker
sudo systemctl restart docker

# Check firewall
sudo ufw status
```

### 2. Performance Optimization

#### Multi-GPU Setup

```bash
# Update docker-compose.production.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

#### Memory Optimization

```bash
# Use mixed precision training
# Already enabled in production Dockerfile

# Optimize data loading
# Use num_workers=4 in DataLoader
```

### 3. Log Analysis

```bash
# View error logs
docker logs architectural-research-production 2>&1 | grep ERROR

# View warning logs
docker logs architectural-research-production 2>&1 | grep WARNING

# Monitor resource usage
docker stats architectural-research-production
```

## 📈 Performance Benchmarks

### Expected Performance

| Model | Parameters | Expected Accuracy | Training Time |
|-------|------------|------------------|---------------|
| ResNet-50 | 23.6M | 8-12% | 2-4 hours |
| Vision Transformer | 85.8M | 8-15% | 4-8 hours |
| Hierarchical | 105M | 4-10% | 8-16 hours |

### Scaling Guidelines

- **Small Scale**: 1 GPU, 100 trials, 10 epochs
- **Medium Scale**: 2-4 GPUs, 500 trials, 20 epochs
- **Large Scale**: 8+ GPUs, 1000+ trials, 50+ epochs

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs: `docker logs architectural-research-production`
3. Check system resources: `./monitor.sh`
4. Verify configuration files
5. Test with smaller experiments first

## 🎯 Next Steps

1. **Deploy to Cloud Server**: Follow the deployment guide
2. **Run Hyperparameter Optimization**: Use Optuna and Ray Tune
3. **Scale Experiments**: Run large-scale training
4. **Monitor Progress**: Use TensorBoard and monitoring tools
5. **Analyze Results**: Review the generated reports
6. **Iterate and Improve**: Based on results, refine the approach

---

**Happy Researching! 🚀**
