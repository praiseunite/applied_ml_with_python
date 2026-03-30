# verify_setup.py
# Run this script to verify your AMLP course environment setup

import sys

print(f"Python Version: {sys.version.split(' ')[0]}")
print(f"Python Executable: {sys.executable}")
print()

# Dictionary mapping common package names to their import names
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
        print(f"  [ OK ] {name:20s} -> {version}")
    except ImportError:
        print(f"  [FAIL] {name:20s} -> NOT INSTALLED")
        all_good = False
    except Exception as e:
        print(f"  [ERR ] {name:20s} -> ERROR: {str(e)}")
        all_good = False

print("-" * 50)
if all_good:
    print("\n🎉 All packages installed successfully! You're ready to start the course.")
else:
    print("\n⚠️  Some packages are missing or failed to import.")
    print("Please activate your virtual environment and run: pip install -r requirements.txt")
