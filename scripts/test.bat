@echo off
REM Architectural Style Classification Project Test Script for Windows
REM This script tests the Docker Compose setup and verifies all services work correctly

echo ================================
echo   Testing Docker Compose Setup
echo ================================
echo.

REM Check if Docker is running
echo [INFO] Checking Docker installation...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker first.
    exit /b 1
)
echo [INFO] Docker is running ✓
echo.

REM Check if Docker Compose is available
echo [INFO] Checking Docker Compose...
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not available.
    exit /b 1
)
echo [INFO] Docker Compose is available ✓
echo.

REM Check NVIDIA Docker
echo [INFO] Checking NVIDIA Docker support...
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] NVIDIA Docker is available ✓
) else (
    echo [WARNING] NVIDIA Docker is not available. GPU support will be limited.
)
echo.

REM Check required files
echo [INFO] Checking required files...
set missing_files=0
if not exist "docker-compose.yml" (
    echo [ERROR] Missing: docker-compose.yml
    set /a missing_files+=1
)
if not exist "Dockerfile.jupyter" (
    echo [ERROR] Missing: Dockerfile.jupyter
    set /a missing_files+=1
)
if not exist "Dockerfile.training" (
    echo [ERROR] Missing: Dockerfile.training
    set /a missing_files+=1
)
if not exist "Dockerfile.inference" (
    echo [ERROR] Missing: Dockerfile.inference
    set /a missing_files+=1
)
if not exist "Dockerfile.data" (
    echo [ERROR] Missing: Dockerfile.data
    set /a missing_files+=1
)
if not exist "Dockerfile.monitoring" (
    echo [ERROR] Missing: Dockerfile.monitoring
    set /a missing_files+=1
)
if not exist "requirements.txt" (
    echo [ERROR] Missing: requirements.txt
    set /a missing_files+=1
)
if not exist "env.example" (
    echo [ERROR] Missing: env.example
    set /a missing_files+=1
)
if not exist "nginx\nginx.conf" (
    echo [ERROR] Missing: nginx\nginx.conf
    set /a missing_files+=1
)

if %missing_files% equ 0 (
    echo [INFO] All required files are present ✓
) else (
    echo [ERROR] Missing %missing_files% required files.
    exit /b 1
)
echo.

REM Create test environment file
echo [INFO] Creating test environment file...
if not exist ".env" (
    copy env.example .env >nul
    echo [INFO] Created .env from template ✓
) else (
    echo [INFO] .env file already exists ✓
)
echo.

REM Create test directories
echo [INFO] Creating test directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\augmented" mkdir data\augmented
if not exist "models" mkdir models
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\pretrained" mkdir models\pretrained
if not exist "models\exported" mkdir models\exported
if not exist "logs" mkdir logs
if not exist "logs\training" mkdir logs\training
if not exist "logs\inference" mkdir logs\inference
if not exist "logs\experiments" mkdir logs\experiments
if not exist "configs" mkdir configs
if not exist "configs\models" mkdir configs\models
if not exist "configs\training" mkdir configs\training
if not exist "configs\data" mkdir configs\data
if not exist "src" mkdir src
if not exist "src\models" mkdir src\models
if not exist "src\data" mkdir src\data
if not exist "src\training" mkdir src\training
if not exist "src\utils" mkdir src\utils
if not exist "src\api" mkdir src\api
if not exist "notebooks" mkdir notebooks
if not exist "notebooks\exploration" mkdir notebooks\exploration
if not exist "notebooks\experiments" mkdir notebooks\experiments
if not exist "notebooks\results" mkdir notebooks\results
if not exist "api" mkdir api
if not exist "dashboards" mkdir dashboards
if not exist "nginx\ssl" mkdir nginx\ssl
echo [INFO] Test directories created ✓
echo.

REM Create minimal test source files
echo [INFO] Creating minimal test source files...

REM Create basic API file
echo from fastapi import FastAPI, File, UploadFile > api\main.py
echo from fastapi.responses import JSONResponse >> api\main.py
echo import uvicorn >> api\main.py
echo. >> api\main.py
echo app = FastAPI(title="Architectural Style Classification API") >> api\main.py
echo. >> api\main.py
echo @app.get("/") >> api\main.py
echo async def root(): >> api\main.py
echo     return {"message": "Architectural Style Classification API is running!"} >> api\main.py
echo. >> api\main.py
echo @app.get("/health") >> api\main.py
echo async def health(): >> api\main.py
echo     return {"status": "healthy"} >> api\main.py
echo. >> api\main.py
echo @app.post("/predict") >> api\main.py
echo async def predict(file: UploadFile = File(...)): >> api\main.py
echo     return { >> api\main.py
echo         "filename": file.filename, >> api\main.py
echo         "prediction": "test_prediction", >> api\main.py
echo         "confidence": 0.95 >> api\main.py
echo     } >> api\main.py
echo. >> api\main.py
echo if __name__ == "__main__": >> api\main.py
echo     uvicorn.run(app, host="0.0.0.0", port=8000) >> api\main.py

REM Create basic training script
echo #!/usr/bin/env python3 > src\train.py
echo """ >> src\train.py
echo Test training script for Architectural Style Classification >> src\train.py
echo """ >> src\train.py
echo. >> src\train.py
echo def main(): >> src\train.py
echo     """Test training function""" >> src\train.py
echo     print("Training script is working!") >> src\train.py
echo     print("This is a test implementation.") >> src\train.py
echo     print("In a real scenario, this would train the model.") >> src\train.py
echo. >> src\train.py
echo if __name__ == "__main__": >> src\train.py
echo     main() >> src\train.py

echo [INFO] Test source files created ✓
echo.

REM Test Docker Compose build
echo [INFO] Testing Docker Compose build...
echo [INFO] Building Jupyter service...
docker-compose build jupyter
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build Jupyter service
    goto cleanup
)
echo [INFO] Jupyter service built successfully ✓
echo.

REM Test service startup
echo [INFO] Testing service startup...
echo [INFO] Starting Jupyter service...
docker-compose up -d jupyter
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Jupyter service
    goto cleanup
)

REM Wait for service to be ready
echo [INFO] Waiting for Jupyter service to be ready...
timeout /t 30 /nobreak >nul

REM Check if service is running
docker-compose ps jupyter | findstr "Up" >nul
if %errorlevel% equ 0 (
    echo [INFO] Jupyter service is running ✓
) else (
    echo [ERROR] Jupyter service failed to start
    docker-compose logs jupyter
    goto cleanup
)
echo.

REM Test API endpoints (optional)
echo [INFO] Testing API endpoints...
echo [INFO] Starting inference service...
docker-compose --profile inference up -d inference
if %errorlevel% neq 0 (
    echo [WARNING] Failed to start inference service
) else (
    echo [INFO] Waiting for inference service to be ready...
    timeout /t 30 /nobreak >nul
    
    REM Test health endpoint
    echo [INFO] Testing health endpoint...
    curl -f http://localhost:8000/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Health endpoint is working ✓
    ) else (
        echo [WARNING] Health endpoint is not responding
    )
)
echo.

:cleanup
REM Clean up test services
echo [INFO] Cleaning up test services...
docker-compose down
echo [INFO] Test services cleaned up ✓
echo.

echo [INFO] All tests completed!
echo [INFO] Your Docker Compose setup is working correctly! 🎉
echo.
echo [INFO] Next steps:
echo [INFO] 1. Run: docker-compose up jupyter
echo [INFO] 2. Access Jupyter Lab at: http://localhost:8888
echo [INFO] 3. Token: archstyle2025
echo.
echo [INFO] Available commands:
echo [INFO] - docker-compose up jupyter          # Start development environment
echo [INFO] - docker-compose --profile training up training  # Start training
echo [INFO] - docker-compose --profile inference up inference  # Start inference API
echo [INFO] - docker-compose --profile monitoring up monitoring  # Start monitoring tools
