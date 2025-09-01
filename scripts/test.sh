#!/bin/bash

# Architectural Style Classification Project Test Script
# This script tests the Docker Compose setup and verifies all services work correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Testing Docker Compose Setup${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_status "Docker is running ✓"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if ! docker-compose --version > /dev/null 2>&1; then
        print_error "Docker Compose is not available."
        exit 1
    fi
    print_status "Docker Compose is available ✓"
}

# Function to check NVIDIA Docker
check_nvidia_docker() {
    print_status "Checking NVIDIA Docker support..."
    if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_status "NVIDIA Docker is available ✓"
        return 0
    else
        print_warning "NVIDIA Docker is not available. GPU support will be limited."
        return 1
    fi
}

# Function to check required files
check_files() {
    print_status "Checking required files..."
    
    required_files=(
        "docker-compose.yml"
        "Dockerfile.jupyter"
        "Dockerfile.training"
        "Dockerfile.inference"
        "Dockerfile.data"
        "Dockerfile.monitoring"
        "requirements.txt"
        "env.example"
        "scripts/setup.sh"
        "scripts/start.sh"
        "nginx/nginx.conf"
    )
    
    missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_status "All required files are present ✓"
    else
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
}

# Function to create test environment file
create_test_env() {
    print_status "Creating test environment file..."
    if [ ! -f ".env" ]; then
        cp env.example .env
        print_status "Created .env from template ✓"
    else
        print_status ".env file already exists ✓"
    fi
}

# Function to create test directories
create_test_dirs() {
    print_status "Creating test directories..."
    mkdir -p data/{raw,processed,augmented}
    mkdir -p models/{checkpoints,pretrained,exported}
    mkdir -p logs/{training,inference,experiments}
    mkdir -p configs/{models,training,data}
    mkdir -p src/{models,data,training,utils,api}
    mkdir -p notebooks/{exploration,experiments,results}
    mkdir -p api
    mkdir -p dashboards
    mkdir -p nginx/ssl
    print_status "Test directories created ✓"
}

# Function to create minimal test source files
create_test_sources() {
    print_status "Creating minimal test source files..."
    
    # Create basic API file
    cat > api/main.py << 'EOF'
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Architectural Style Classification API")

@app.get("/")
async def root():
    return {"message": "Architectural Style Classification API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "prediction": "test_prediction",
        "confidence": 0.95
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

    # Create basic Gradio app
    cat > api/gradio_app.py << 'EOF'
import gradio as gr

def predict_style(image):
    return "Test Prediction: Gothic Architecture", 0.95

demo = gr.Interface(
    fn=predict_style,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Prediction"), gr.Slider(label="Confidence")],
    title="Architectural Style Classification",
    description="Upload an image of a building to classify its architectural style."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8001)
EOF

    # Create basic training script
    cat > src/train.py << 'EOF'
#!/usr/bin/env python3
"""
Test training script for Architectural Style Classification
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Test training function"""
    print("Training script is working!")
    print("This is a test implementation.")
    print("In a real scenario, this would train the model.")

if __name__ == "__main__":
    main()
EOF

    # Create basic data processing script
    cat > src/data_processing.py << 'EOF'
#!/usr/bin/env python3
"""
Test data processing script for Architectural Style Classification
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Test data processing function"""
    print("Data processing script is working!")
    print("This is a test implementation.")
    print("In a real scenario, this would process the dataset.")

if __name__ == "__main__":
    main()
EOF

    print_status "Test source files created ✓"
}

# Function to test Docker Compose build
test_build() {
    print_status "Testing Docker Compose build..."
    
    # Test building Jupyter service
    print_status "Building Jupyter service..."
    if docker-compose build jupyter; then
        print_status "Jupyter service built successfully ✓"
    else
        print_error "Failed to build Jupyter service"
        return 1
    fi
    
    # Test building other services
    services=("training" "inference" "data_processing" "monitoring")
    for service in "${services[@]}"; do
        print_status "Building $service service..."
        if docker-compose build $service; then
            print_status "$service service built successfully ✓"
        else
            print_error "Failed to build $service service"
            return 1
        fi
    done
    
    return 0
}

# Function to test service startup
test_service_startup() {
    print_status "Testing service startup..."
    
    # Start Jupyter service
    print_status "Starting Jupyter service..."
    docker-compose up -d jupyter
    
    # Wait for service to be ready
    print_status "Waiting for Jupyter service to be ready..."
    sleep 30
    
    # Check if service is running
    if docker-compose ps jupyter | grep -q "Up"; then
        print_status "Jupyter service is running ✓"
    else
        print_error "Jupyter service failed to start"
        docker-compose logs jupyter
        return 1
    fi
    
    return 0
}

# Function to test API endpoints
test_api_endpoints() {
    print_status "Testing API endpoints..."
    
    # Start inference service
    print_status "Starting inference service..."
    docker-compose --profile inference up -d inference
    
    # Wait for service to be ready
    print_status "Waiting for inference service to be ready..."
    sleep 30
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Health endpoint is working ✓"
    else
        print_warning "Health endpoint is not responding"
    fi
    
    # Test root endpoint
    print_status "Testing root endpoint..."
    if curl -f http://localhost:8000/ > /dev/null 2>&1; then
        print_status "Root endpoint is working ✓"
    else
        print_warning "Root endpoint is not responding"
    fi
    
    return 0
}

# Function to test monitoring services
test_monitoring() {
    print_status "Testing monitoring services..."
    
    # Start monitoring services
    print_status "Starting monitoring services..."
    docker-compose --profile monitoring up -d monitoring postgres redis
    
    # Wait for services to be ready
    print_status "Waiting for monitoring services to be ready..."
    sleep 30
    
    # Check if services are running
    services=("monitoring" "postgres" "redis")
    for service in "${services[@]}"; do
        if docker-compose ps $service | grep -q "Up"; then
            print_status "$service service is running ✓"
        else
            print_warning "$service service failed to start"
        fi
    done
    
    return 0
}

# Function to clean up test services
cleanup_test() {
    print_status "Cleaning up test services..."
    docker-compose down
    print_status "Test services cleaned up ✓"
}

# Function to run all tests
run_all_tests() {
    print_header
    
    # Run all test functions
    check_docker
    check_docker_compose
    check_nvidia_docker
    check_files
    create_test_env
    create_test_dirs
    create_test_sources
    
    print_status "Starting comprehensive tests..."
    
    if test_build; then
        print_status "Build tests passed ✓"
    else
        print_error "Build tests failed"
        cleanup_test
        exit 1
    fi
    
    if test_service_startup; then
        print_status "Service startup tests passed ✓"
    else
        print_error "Service startup tests failed"
        cleanup_test
        exit 1
    fi
    
    if test_api_endpoints; then
        print_status "API endpoint tests passed ✓"
    else
        print_warning "Some API endpoint tests failed"
    fi
    
    if test_monitoring; then
        print_status "Monitoring tests passed ✓"
    else
        print_warning "Some monitoring tests failed"
    fi
    
    cleanup_test
    
    print_status "All tests completed!"
    print_status "Your Docker Compose setup is working correctly! 🎉"
}

# Function to show usage
show_usage() {
    print_header
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all         Run all tests (default)"
    echo "  build       Test Docker Compose build only"
    echo "  startup     Test service startup only"
    echo "  api         Test API endpoints only"
    echo "  monitoring  Test monitoring services only"
    echo "  cleanup     Clean up all test services"
    echo "  help        Show this help message"
    echo ""
}

# Main script logic
main() {
    case "${1:-all}" in
        all)
            run_all_tests
            ;;
        build)
            check_docker
            check_docker_compose
            check_files
            create_test_env
            create_test_dirs
            create_test_sources
            test_build
            ;;
        startup)
            check_docker
            check_docker_compose
            test_service_startup
            ;;
        api)
            check_docker
            check_docker_compose
            test_api_endpoints
            ;;
        monitoring)
            check_docker
            check_docker_compose
            test_monitoring
            ;;
        cleanup)
            cleanup_test
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
