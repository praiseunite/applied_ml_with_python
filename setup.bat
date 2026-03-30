@echo off
echo =======================================================
echo setup.bat: Automated Environment Setup for AMLP Course
echo =======================================================
echo.

echo [1/4] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)
python --version

echo.
echo [2/4] Setting up Virtual Environment...
if exist venv\Scripts\activate (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo [3/4] Installing / Updating Dependencies...
call venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo Installing requirements (this might take a few minutes)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed to install. 
    echo Please check the error messages above.
) else (
    echo Dependencies installed successfully.
)

echo.
echo [4/4] Verifying Setup...
if exist verify_setup.py (
    python verify_setup.py
) else (
    echo [WARNING] verify_setup.py not found.
)

echo.
echo =======================================================
echo Setup Complete! 
echo.
echo IMPORTANT: To start working on the course, you MUST activate 
echo the virtual environment first by running:
echo.
echo     venv\Scripts\activate
echo.
echo =======================================================
pause
