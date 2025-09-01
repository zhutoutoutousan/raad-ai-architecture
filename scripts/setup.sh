#!/bin/bash

# Architectural Style Classification Project Setup Script
# This script sets up the project structure and downloads the dataset

set -e

echo "🚀 Setting up Architectural Style Classification Project..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,augmented}
mkdir -p models/{checkpoints,pretrained,exported}
mkdir -p logs/{training,inference,experiments}
mkdir -p configs/{models,training,data}
mkdir -p src/{models,data,training,utils,api}
mkdir -p notebooks/{exploration,experiments,results}
mkdir -p scripts
mkdir -p api
mkdir -p dashboards
mkdir -p nginx/ssl

# Set up Kaggle credentials
echo "🔑 Setting up Kaggle credentials..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Please create your Kaggle API credentials at https://www.kaggle.com/account"
    echo "Download kaggle.json and place it in ~/.kaggle/"
    read -p "Press Enter when you have placed kaggle.json in ~/.kaggle/"
fi

# Download dataset
echo "📥 Downloading architecture dataset..."
if command -v kaggle &> /dev/null; then
    kaggle datasets download -d wwymak/architecture-dataset -p data/raw --unzip
    echo "✅ Dataset downloaded successfully!"
else
    echo "⚠️  Kaggle CLI not found. Please install it or download the dataset manually."
    echo "Dataset URL: https://www.kaggle.com/datasets/wwymak/architecture-dataset"
fi

# Create initial configuration files
echo "⚙️  Creating configuration files..."

# Model configuration
cat > configs/models/default.yaml << EOF
model:
  backbone: efficientnet-b4
  num_classes: 25
  pretrained: true
  dropout: 0.5
  
hierarchical:
  enabled: true
  levels:
    - name: broad
      classes: [ancient, medieval, renaissance, modern, contemporary]
    - name: fine
      classes: [greek, roman, gothic, romanesque, baroque, art_deco, bauhaus, international, postmodern, deconstructivism]

attention:
  enabled: true
  heads: 8
  dim: 768
EOF

# Training configuration
cat > configs/training/default.yaml << EOF
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  optimizer: adamw
  scheduler: cosine
  
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  image_size: [224, 224]
  augmentations:
    - horizontal_flip
    - rotation
    - brightness_contrast
    - gaussian_noise

gpu:
  mixed_precision: true
  gradient_accumulation_steps: 1
  memory_fraction: 0.9
EOF

# Data configuration
cat > configs/data/default.yaml << EOF
dataset:
  name: architecture_style
  path: data/raw
  classes: 25
  
augmentation:
  style_preserving: true
  geometric:
    - perspective_transform
    - rotation_with_constraints
    - scale_with_aspect_ratio
  lighting:
    - brightness_variation
    - contrast_adjustment
    - shadow_simulation
EOF

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/
data/processed/
data/augmented/
*.csv
*.json
*.pkl
*.h5

# Models
models/checkpoints/
models/pretrained/
models/exported/
*.pth
*.pt
*.onnx
*.pb

# Logs
logs/
*.log

# Environment
.env
.env.local

# Docker
.dockerignore

# OS
.DS_Store
Thumbs.db

# Kaggle
.kaggle/

# Weights & Biases
wandb/

# MLflow
mlruns/
mlflow.db
EOF

# Create initial source files
echo "📝 Creating initial source files..."

# Main training script
cat > src/train.py << 'EOF'
#!/usr/bin/env python3
"""
Main training script for Architectural Style Classification
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from training.trainer import ArchitecturalStyleTrainer
from data.dataset import ArchitecturalStyleDataset
from models.hierarchical_classifier import HierarchicalArchitecturalClassifier
from utils.config import load_config
from utils.logging import setup_logging

def main():
    """Main training function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    
    # Initialize model
    model = HierarchicalArchitecturalClassifier(config)
    
    # Initialize dataset
    dataset = ArchitecturalStyleDataset(config)
    
    # Initialize trainer
    trainer = ArchitecturalStyleTrainer(model, dataset, config)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
EOF

# Make scripts executable
chmod +x scripts/*.sh

echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Copy env.example to .env and update with your API keys"
echo "2. Run: docker-compose up jupyter"
echo "3. Access Jupyter Lab at: http://localhost:8888"
echo "4. Token: archstyle2025"
echo ""
echo "🎯 Available commands:"
echo "- docker-compose up jupyter          # Start development environment"
echo "- docker-compose --profile training up training  # Start training"
echo "- docker-compose --profile inference up inference  # Start inference API"
echo "- docker-compose --profile monitoring up monitoring  # Start monitoring tools"
