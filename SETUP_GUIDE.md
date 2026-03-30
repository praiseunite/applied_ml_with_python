# 🛠️ Environment Setup Guide

This guide walks you through setting up your development environment for the Applied Machine Learning Using Python course.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Installation](#python-installation)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Installing Dependencies](#installing-dependencies)
5. [Jupyter Notebook Setup](#jupyter-notebook-setup)
6. [VS Code Setup (Recommended)](#vs-code-setup)
7. [Hugging Face Account Setup](#hugging-face-account-setup)
8. [Verifying Your Setup](#verifying-your-setup)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 / macOS 10.15 / Ubuntu 20.04 | Windows 11 / macOS 13+ / Ubuntu 22.04 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB free | 20 GB free |
| **Python** | 3.10 | 3.11 or 3.12 |
| **Internet** | Required for dataset downloads and deployment | Stable broadband |

---

## Python Installation

### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **⚠️ IMPORTANT**: Check ✅ "Add Python to PATH" during installation
3. Open Command Prompt and verify:

```bash
python --version
# Expected: Python 3.10.x or higher
```

### macOS

```bash
# Using Homebrew (recommended)
brew install python@3.11

# Verify
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3 --version
```

---

## 🚀 Quick Start (Automated Setup)

The absolute easiest way to set up your isolated course environment without breaking your global Python is to use the provided setup scripts. This script automatically creates a **Virtual Environment (`venv`)**, activates it, and installs all the packages securely from `requirements.txt`.

### For Windows:
Simply double-click `setup.bat` in the root folder, OR run it from your command prompt:
```cmd
setup.bat
```

### For macOS / Linux:
Open your terminal and run:
```bash
chmod +x setup.sh  # Make script executable
./setup.sh
```

**⚠️ CRITICAL AFTER SETUP**: Every time you open a new terminal or command prompt to work on the course, you **MUST** activate the virtual environment!
- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

---

## Manual Setup Method (Optional)

If the automated scripts fail, you can set up the isolated environment manually:

### 1. Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
# You should see (venv) in your terminal prompt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

With your virtual environment activated:
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all course dependencies securely inside the venv
pip install -r requirements.txt
```

This will run for 5-10 minutes. If you encounter issues with giant packages (like TensorFlow), try installing them sequentially:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install xgboost lightgbm imbalanced-learn
pip install tensorflow
pip install flask gradio streamlit
```

---

## Jupyter Notebook Setup

### Option 1: Classic Jupyter Notebook

```bash
# Already installed via requirements.txt
jupyter notebook
```

This opens Jupyter in your browser at `http://localhost:8888`

### Option 2: JupyterLab (Modern Interface)

```bash
pip install jupyterlab
jupyter lab
```

### Running Course Notebooks

1. Activate your virtual environment
2. Navigate to the course directory
3. Run `jupyter notebook`
4. Open any session folder → `notebooks/` → click the `.ipynb` file
5. Run cells with `Shift + Enter`

---

## VS Code Setup

VS Code is the recommended editor for this course.

### Installation

1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install these extensions:
   - **Python** (Microsoft) — Python language support
   - **Jupyter** (Microsoft) — Run notebooks in VS Code
   - **Pylance** (Microsoft) — Python IntelliSense
   - **GitLens** — Enhanced Git integration

### Configure Python Interpreter

1. Open VS Code in the course directory
2. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose the `venv` interpreter from this project

### Running Notebooks in VS Code

1. Open any `.ipynb` file
2. Select the `venv` kernel when prompted
3. Run cells with `Shift + Enter`

---

## Hugging Face Account Setup

You'll need a Hugging Face account to deploy your portfolio projects.

### Step 1: Create Account

1. Go to [huggingface.co](https://huggingface.co/)
2. Click "Sign Up"
3. Complete registration (free account is sufficient)

### Step 2: Create Access Token

1. Go to Settings → Access Tokens → New Token
2. Name it: `amlp-course`
3. Type: `Write`
4. Copy and save the token securely

### Step 3: Install Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

### Step 4: Verify

```bash
huggingface-cli whoami
# Should display your username
```

### Step 5: Deploy Your First Space (Test)

```bash
# We'll do this properly in Session 8, but you can test:
pip install gradio
```

```python
# test_deploy.py
import gradio as gr

def greet(name):
    return f"Hello {name}! Welcome to Applied ML with Python 🧠"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

---

## Verifying Your Setup

Run this verification script to make sure everything is installed correctly:

```python
# verify_setup.py — Run this to check your environment
import sys

print(f"Python Version: {sys.version}")
print(f"Python Path: {sys.executable}")
print()

packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tensorflow": "tensorflow",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "imbalanced-learn": "imblearn",
    "flask": "flask",
    "gradio": "gradio",
    "plotly": "plotly",
    "shap": "shap",
    "statsmodels": "statsmodels",
    "prophet": "prophet",
    "gymnasium": "gymnasium",
    "joblib": "joblib",
    "jupyter": "jupyter_core",
}

print("Package Verification:")
print("-" * 50)
all_good = True
for name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        print(f"  ✅ {name:25s} → {version}")
    except ImportError:
        print(f"  ❌ {name:25s} → NOT INSTALLED")
        all_good = False

print("-" * 50)
if all_good:
    print("\n🎉 All packages installed successfully! You're ready to start the course.")
else:
    print("\n⚠️  Some packages are missing. Run: pip install -r requirements.txt")
```

Save this as `verify_setup.py` and run:

```bash
python verify_setup.py
```

---

## Troubleshooting

### "pip is not recognized"

```bash
# Windows: Use python -m pip instead
python -m pip install -r requirements.txt
```

### TensorFlow installation fails

```bash
# Make sure you have Python 3.10-3.12 (TensorFlow doesn't support 3.13+ yet)
python --version

# Try installing specific version
pip install tensorflow==2.15.0
```

### Prophet installation fails

```bash
# Prophet requires additional dependencies on some systems
# Windows:
pip install pystan==2.19.1.1
pip install prophet

# macOS:
brew install gcc
pip install prophet
```

### "No module named 'gymnasium'"

```bash
# gymnasium replaced old 'gym' package
pip install gymnasium[classic_control]
```

### Jupyter kernel not showing

```bash
# Register your venv as a Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=amlp --display-name="AMLP Course"
```

### General: Nuclear Option

If nothing works, start fresh:

```bash
# Delete and recreate virtual environment
deactivate
rmdir /s /q venv          # Windows
rm -rf venv               # macOS/Linux

python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

---

## Need Help?

- 📧 Contact your instructor
- 📚 Check the [Python docs](https://docs.python.org/3/)
- 💬 Ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/python)
- 🤗 Hugging Face [documentation](https://huggingface.co/docs)

---

*Next Step: Open [Session 01](../Session_01_Introduction_to_ML/) and start learning!*
