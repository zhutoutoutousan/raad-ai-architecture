#!/bin/bash

# Cloud Deployment Script for Architectural Style Classification Research
# This script sets up the research environment on a cloud server

set -e

echo "🚀 Cloud Deployment Script for Architectural Research"
echo "=================================================="

# Configuration
DOCKER_IMAGE_NAME="architectural-research"
DOCKER_TAG="latest"
DOCKER_HUB_USERNAME="your-username"  # Change this to your Docker Hub username

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check Docker installation
print_status "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose are installed"

# Check NVIDIA Docker support
print_status "Checking NVIDIA Docker support..."
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    print_warning "NVIDIA Docker support not detected. GPU acceleration may not work."
    print_warning "Install nvidia-docker2 for GPU support: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
else
    print_success "NVIDIA Docker support detected"
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data models logs results paper configs ray_results
print_success "Directories created"

# Build Docker image
print_status "Building Docker image..."
docker build -f Dockerfile.production -t ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} .
print_success "Docker image built successfully"

# Tag for Docker Hub
print_status "Tagging image for Docker Hub..."
docker tag ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG}
print_success "Image tagged for Docker Hub"

# Create environment file
print_status "Creating environment file..."
cat > .env << EOF
# Docker Hub credentials (optional)
DOCKER_HUB_USERNAME=${DOCKER_HUB_USERNAME}

# Weights & Biases (optional)
WANDB_API_KEY=

# Kaggle credentials (optional)
KAGGLE_USERNAME=
KAGGLE_KEY=

# GPU configuration
CUDA_VISIBLE_DEVICES=0
EOF
print_success "Environment file created"

# Create deployment script
print_status "Creating deployment script..."
cat > deploy.sh << 'EOF'
#!/bin/bash

# Quick deployment script
echo "🚀 Deploying Architectural Research Environment..."

# Pull latest image
docker pull ${DOCKER_HUB_USERNAME}/architectural-research:latest

# Start services
docker-compose -f docker-compose.production.yml up -d

echo "✅ Deployment completed!"
echo "📊 TensorBoard: http://localhost:6006"
echo "🚀 Ray Dashboard: http://localhost:8265"
echo "📁 Results: ./results/"
echo "📈 Logs: ./logs/"
EOF

chmod +x deploy.sh
print_success "Deployment script created"

# Create cloud server setup script
print_status "Creating cloud server setup script..."
cat > cloud_setup.sh << 'EOF'
#!/bin/bash

# Cloud Server Setup Script
echo "☁️  Cloud Server Setup for Architectural Research"

# Update system
sudo apt-get update
sudo apt-get upgrade -y

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

echo "✅ Cloud server setup completed!"
echo "🔑 Please log out and log back in for Docker group changes to take effect"
EOF

chmod +x cloud_setup.sh
print_success "Cloud server setup script created"

# Create scaling script
print_status "Creating scaling script..."
cat > scale_experiments.sh << 'EOF'
#!/bin/bash

# Scaling Experiments Script
echo "📈 Scaling Architectural Research Experiments"

# Run hyperparameter optimization
echo "🔍 Running hyperparameter optimization..."
docker exec architectural-research-production python hyperparameter_optimization.py --method optuna --trials 100 --epochs 20

# Run scaled experiments
echo "🚀 Running scaled experiments..."
docker exec architectural-research-production python scale_up_experiments.py --experiment hierarchical_advanced --epochs 100 --batch_size 32 --samples_per_class 1000

# Run baseline comparisons
echo "📊 Running baseline comparisons..."
docker exec architectural-research-production python run_experiment.py --experiment baseline_resnet --epochs 50 --batch_size 64 --data_path /app/data/real_architecture/processed
docker exec architectural-research-production python run_experiment.py --experiment baseline_vit --epochs 50 --batch_size 32 --data_path /app/data/real_architecture/processed

echo "✅ Scaling experiments completed!"
EOF

chmod +x scale_experiments.sh
print_success "Scaling script created"

# Create monitoring script
print_status "Creating monitoring script..."
cat > monitor.sh << 'EOF'
#!/bin/bash

# Monitoring Script
echo "📊 Monitoring Architectural Research"

echo "🐳 Container Status:"
docker ps

echo ""
echo "📈 GPU Usage:"
nvidia-smi

echo ""
echo "💾 Disk Usage:"
df -h

echo ""
echo "📊 Recent Logs:"
docker logs --tail 20 architectural-research-production

echo ""
echo "📁 Results Directory:"
ls -la results/

echo ""
echo "📈 Logs Directory:"
ls -la logs/
EOF

chmod +x monitor.sh
print_success "Monitoring script created"

# Push to Docker Hub (optional)
read -p "🤔 Do you want to push the image to Docker Hub? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Pushing image to Docker Hub..."
    docker push ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG}
    print_success "Image pushed to Docker Hub"
fi

# Final instructions
echo ""
echo "🎉 Cloud Deployment Setup Completed!"
echo "=================================="
echo ""
echo "📋 Next Steps:"
echo "1. Upload this directory to your cloud server"
echo "2. Run: ./cloud_setup.sh (on cloud server)"
echo "3. Run: ./deploy.sh (to start services)"
echo "4. Run: ./scale_experiments.sh (to start experiments)"
echo "5. Run: ./monitor.sh (to monitor progress)"
echo ""
echo "🌐 Access Points:"
echo "- TensorBoard: http://YOUR_SERVER_IP:6006"
echo "- Ray Dashboard: http://YOUR_SERVER_IP:8265"
echo ""
echo "📁 Important Files:"
echo "- docker-compose.production.yml: Production deployment"
echo "- hyperparameter_optimization.py: HPO framework"
echo "- scale_up_experiments.py: Large-scale experiments"
echo "- .env: Environment configuration"
echo ""
echo "🔧 Configuration:"
echo "- Edit .env file to add API keys"
echo "- Modify docker-compose.production.yml for your setup"
echo "- Adjust hyperparameter_optimization.py for your needs"
echo ""
print_success "Ready for cloud deployment! 🚀"
