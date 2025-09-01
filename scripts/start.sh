#!/bin/bash

# Architectural Style Classification Project Start Script
# This script provides easy access to different services

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
    echo -e "${BLUE}  Architectural Style Project${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if NVIDIA Docker is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_warning "NVIDIA Docker runtime not available. GPU support may be limited."
    else
        print_status "NVIDIA Docker runtime is available."
    fi
}

# Function to show usage
show_usage() {
    print_header
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  jupyter     Start Jupyter Lab development environment"
    echo "  training    Start training service"
    echo "  inference   Start inference API service"
    echo "  monitoring  Start monitoring tools (TensorBoard, MLflow, Grafana)"
    echo "  data        Start data processing service"
    echo "  all         Start all services"
    echo "  stop        Stop all services"
    echo "  logs        Show logs for all services"
    echo "  status      Show status of all services"
    echo "  clean       Clean up containers and volumes"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 jupyter     # Start development environment"
    echo "  $0 training    # Start model training"
    echo "  $0 all         # Start all services"
    echo ""
}

# Function to start Jupyter
start_jupyter() {
    print_status "Starting Jupyter Lab development environment..."
    docker-compose up jupyter
}

# Function to start training
start_training() {
    print_status "Starting training service..."
    docker-compose --profile training up training
}

# Function to start inference
start_inference() {
    print_status "Starting inference API service..."
    docker-compose --profile inference up inference
}

# Function to start monitoring
start_monitoring() {
    print_status "Starting monitoring tools..."
    docker-compose --profile monitoring up monitoring
}

# Function to start data processing
start_data() {
    print_status "Starting data processing service..."
    docker-compose --profile data up data_processing
}

# Function to start all services
start_all() {
    print_status "Starting all services..."
    docker-compose --profile training --profile inference --profile monitoring --profile data up -d
    print_status "All services started!"
    print_status "Access points:"
    print_status "  - Jupyter Lab: http://localhost:8888 (token: archstyle2025)"
    print_status "  - FastAPI: http://localhost:8000"
    print_status "  - Gradio UI: http://localhost:8001"
    print_status "  - TensorBoard: http://localhost:6006"
    print_status "  - MLflow: http://localhost:5000"
    print_status "  - Grafana: http://localhost:3000"
}

# Function to stop all services
stop_all() {
    print_status "Stopping all services..."
    docker-compose down
    print_status "All services stopped!"
}

# Function to show logs
show_logs() {
    print_status "Showing logs for all services..."
    docker-compose logs -f
}

# Function to show status
show_status() {
    print_status "Service status:"
    docker-compose ps
}

# Function to clean up
clean_up() {
    print_warning "This will remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_status "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to check GPU status
check_gpu() {
    print_status "Checking GPU status..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        print_warning "nvidia-smi not found. GPU status cannot be displayed."
    fi
}

# Main script logic
main() {
    # Check prerequisites
    check_docker
    check_nvidia_docker
    
    # Parse command line arguments
    case "${1:-help}" in
        jupyter)
            start_jupyter
            ;;
        training)
            start_training
            ;;
        inference)
            start_inference
            ;;
        monitoring)
            start_monitoring
            ;;
        data)
            start_data
            ;;
        all)
            start_all
            ;;
        stop)
            stop_all
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        clean)
            clean_up
            ;;
        gpu)
            check_gpu
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
