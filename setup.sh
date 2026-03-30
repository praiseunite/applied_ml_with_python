#!/bin/bash
echo "======================================================="
echo "setup.sh: Automated Environment Setup for AMLP Course"
echo "======================================================="
echo ""

# Exit immediately if a command exits with a non-zero status
set -e

echo "[1/4] Checking Python version..."
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python is not installed."
    echo "Please install Python 3.10+."
    exit 1
fi
$PYTHON_CMD --version

echo ""
echo "[2/4] Setting up Virtual Environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment (venv)..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        echo "If you are on Ubuntu/Debian, you may need to run: sudo apt install python3-venv"
        exit 1
    fi
fi

echo ""
echo "[3/4] Installing / Updating Dependencies..."
source venv/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip >/dev/null 2>&1
echo "Installing requirements (this might take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "[4/4] Verifying Setup..."
if [ -f "verify_setup.py" ]; then
    python verify_setup.py
else
    echo "[WARNING] verify_setup.py not found."
fi

echo ""
echo "======================================================="
echo "Setup Complete!"
echo ""
echo "IMPORTANT: To start working on the course, you MUST activate"
echo "the virtual environment first by running:"
echo ""
echo "    source venv/bin/activate"
echo ""
echo "======================================================="
