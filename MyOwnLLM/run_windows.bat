@echo off
REM =====================================================================
REM MyOwnLLM - Windows Automation Batch Script
REM =====================================================================
title MyOwnLLM Setup and Runner
color 0A

echo =====================================================================
echo                 Welcome to MyOwnLLM for Windows!
echo     Building a Decoder-Only Transformer from Scratch on CPU
echo =====================================================================
echo.

REM 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    echo Make sure to check the box "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python is installed:
python --version
echo.

REM 2. Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating Python virtual environment in .env ...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [OK] Virtual environment already exists.
)

REM 3. Activate virtual environment
call venv\Scripts\activate.bat

REM 4. Install dependencies if torch is not installed
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] Installing PyTorch CPU edition (optimized for Windows laptop)...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install PyTorch.
        pause
        exit /b 1
    )
    echo [OK] PyTorch installed successfully.
) else (
    echo [OK] PyTorch is already installed.
)

echo.
echo =====================================================================
echo Step 1: Running Quick Smoke Test...
echo =====================================================================
python test_smoke.py
if %errorlevel% neq 0 (
    echo [ERROR] Smoke test failed! Please check error output above.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo Menu: What would you like to do?
echo =====================================================================
echo [1] Train model on dataset.txt (takes ~2 minutes on CPU)
echo [2] Generate text using existing checkpoint (my_own_llm.pt)
echo [3] Exit
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" (
    echo.
    echo Starting training (20 epochs)...
    python train.py --epochs 20
    echo.
    echo Training complete! Now generating sample text:
    python generate.py --prompt "Once upon a time"
    pause
    exit /b 0
)

if "%choice%"=="2" (
    echo.
    set /p custom_prompt="Enter starting prompt (press Enter for default): "
    if "%custom_prompt%"=="" set custom_prompt=Once upon a time
    python generate.py --prompt "%custom_prompt%"
    pause
    exit /b 0
)

if "%choice%"=="3" (
    exit /b 0
)

echo Invalid choice.
pause
