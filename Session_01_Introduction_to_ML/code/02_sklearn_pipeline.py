"""
============================================================
02_sklearn_pipeline.py
Applied Machine Learning Using Python — Session 01
============================================================
Topic: Production-Grade ML Pipeline with Scikit-Learn
Level: 🟡 Intermediate → 🔴 Advanced

This script demonstrates how real ML code is written in industry:
- sklearn Pipeline for reproducible workflows
- ColumnTransformer for mixed data types
- GridSearchCV for hyperparameter tuning
- Proper train/validation/test splits
- Feature importance analysis
- Model persistence (save & load)

Dataset: California Housing (Real Estate)
- 20,640 samples, 8 features
- Target: Median house value (in $100,000s)
- Reference: Pace & Barry (1997), Sparse Spatial Autoregressions

Usage:
    python 02_sklearn_pipeline.py

Requirements:
    pip install numpy pandas scikit-learn matplotlib seaborn joblib
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def plot_learning_curve(estimator, X, y, title, ax, cv=5):
    """Plot learning curve to diagnose bias/variance."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    
    ax.fill_between(
        train_sizes,
        train_rmse.mean(axis=1) - train_rmse.std(axis=1),
        train_rmse.mean(axis=1) + train_rmse.std(axis=1),
        alpha=0.1, color="#3498db",
    )
    ax.fill_between(
        train_sizes,
        val_rmse.mean(axis=1) - val_rmse.std(axis=1),
        val_rmse.mean(axis=1) + val_rmse.std(axis=1),
        alpha=0.1, color="#e74c3c",
    )
    ax.plot(train_sizes, train_rmse.mean(axis=1), "o-", color="#3498db",
            label="Training RMSE")
    ax.plot(train_sizes, val_rmse.mean(axis=1), "o-", color="#e74c3c",
            label="Validation RMSE")
    ax.set_title(title)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    # ══════════════════════════════════════════════════════════
    # STEP 1: Load & Explore Data
    # ══════════════════════════════════════════════════════════
    print_header("STEP 1: Data Loading & Exploration")

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # Includes features AND target

    print(f"Dataset: California Housing Prices")
    print(f"Samples: {len(df):,}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"\nFeature Descriptions:")
    descriptions = {
        "MedInc": "Median income in block group (tens of thousands)",
        "HouseAge": "Median house age in block group",
        "AveRooms": "Average number of rooms per household",
        "AveBedrms": "Average number of bedrooms per household",
        "Population": "Block group population",
        "AveOccup": "Average number of household members",
        "Latitude": "Block group latitude",
        "Longitude": "Block group longitude",
        "MedHouseVal": "Median house value (target, $100,000s)",
    }
    for col, desc in descriptions.items():
        if col in df.columns:
            print(f"  {col:15s} → {desc}")

    print(f"\nTarget Statistics:")
    print(f"  Min:    ${df['MedHouseVal'].min() * 100000:>12,.0f}")
    print(f"  Mean:   ${df['MedHouseVal'].mean() * 100000:>12,.0f}")
    print(f"  Median: ${df['MedHouseVal'].median() * 100000:>12,.0f}")
    print(f"  Max:    ${df['MedHouseVal'].max() * 100000:>12,.0f}")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Data Splitting Strategy
    # ══════════════════════════════════════════════════════════
    print_header("STEP 2: Data Splitting")

    X = housing.data
    y = housing.target

    # First split: separate test set (held out until final evaluation)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Second split: training and validation from dev set
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.176, random_state=42  # 0.176 of 85% ≈ 15%
    )

    print(f"Training set:   {len(X_train):>6,} samples ({len(X_train)/len(X):.0%})")
    print(f"Validation set: {len(X_val):>6,} samples ({len(X_val)/len(X):.0%})")
    print(f"Test set:       {len(X_test):>6,} samples ({len(X_test)/len(X):.0%})")
    print(f"\n⚠️  Test set is ONLY used for final evaluation — never for tuning!")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Building sklearn Pipelines
    # ══════════════════════════════════════════════════════════
    print_header("STEP 3: Building Pipelines")

    # Pipeline 1: Simple linear model
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])

    # Pipeline 2: Polynomial features + Ridge
    pipe_poly = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", Ridge()),
    ])

    # Pipeline 3: Random Forest (no scaling needed)
    pipe_rf = Pipeline([
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    # Pipeline 4: Gradient Boosting
    pipe_gb = Pipeline([
        ("model", GradientBoostingRegressor(random_state=42)),
    ])

    print("  Built 4 pipelines:")
    print("  1. StandardScaler → Ridge Regression")
    print("  2. StandardScaler → PolynomialFeatures → Ridge")
    print("  3. Random Forest (no scaling needed)")
    print("  4. Gradient Boosting")

    # ══════════════════════════════════════════════════════════
    # STEP 4: Hyperparameter Tuning with GridSearchCV
    # ══════════════════════════════════════════════════════════
    print_header("STEP 4: Hyperparameter Tuning")

    # Define parameter grids
    param_grids = {
        "Ridge": {
            "pipeline": pipe_ridge,
            "params": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "Random Forest": {
            "pipeline": pipe_rf,
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
            },
        },
        "Gradient Boosting": {
            "pipeline": pipe_gb,
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        },
    }

    best_models = {}
    for name, config in param_grids.items():
        print(f"\n  Tuning {name}...")
        grid = GridSearchCV(
            config["pipeline"],
            config["params"],
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        # Evaluate on validation set
        val_pred = grid.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        cv_rmse = np.sqrt(-grid.best_score_)

        best_models[name] = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "cv_rmse": cv_rmse,
        }

        print(f"    Best Params: {grid.best_params_}")
        print(f"    CV RMSE:     {cv_rmse:.4f} (${cv_rmse * 100000:,.0f})")
        print(f"    Val RMSE:    {val_rmse:.4f} (${val_rmse * 100000:,.0f})")
        print(f"    Val R²:      {val_r2:.4f}")

    # ══════════════════════════════════════════════════════════
    # STEP 5: Select Best Model & Final Evaluation
    # ══════════════════════════════════════════════════════════
    print_header("STEP 5: Final Model Selection & Test Evaluation")

    # Select best model based on validation performance
    best_name = min(best_models, key=lambda k: best_models[k]["val_rmse"])
    best_pipeline = best_models[best_name]["model"]
    print(f"  🏆 Best Model: {best_name}")

    # FINAL evaluation on held-out test set
    y_pred_test = best_pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n  ═══ FINAL TEST RESULTS ═══")
    print(f"  Test RMSE: {test_rmse:.4f} (${test_rmse * 100000:,.0f} average error)")
    print(f"  Test MAE:  {test_mae:.4f} (${test_mae * 100000:,.0f})")
    print(f"  Test R²:   {test_r2:.4f}")

    # ══════════════════════════════════════════════════════════
    # STEP 6: Feature Importance Analysis
    # ══════════════════════════════════════════════════════════
    print_header("STEP 6: Feature Importance")

    if hasattr(best_pipeline.named_steps.get("model", best_pipeline), "feature_importances_"):
        model_step = best_pipeline.named_steps["model"]
        importances = pd.Series(
            model_step.feature_importances_,
            index=X.columns,
        ).sort_values(ascending=False)

        print(f"\n  Feature Importances ({best_name}):")
        for feat, imp in importances.items():
            bar = "█" * int(imp * 100)
            print(f"    {feat:15s} {imp:.4f} {bar}")

    # ══════════════════════════════════════════════════════════
    # STEP 7: Visualizations
    # ══════════════════════════════════════════════════════════
    print_header("STEP 7: Result Visualizations")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Production ML Pipeline — California Housing", fontsize=16, fontweight="bold")

    # Plot 1: Model comparison
    model_names = list(best_models.keys())
    val_rmses = [best_models[n]["val_rmse"] for n in model_names]
    colors = ["#2ecc71" if n == best_name else "#3498db" for n in model_names]
    axes[0, 0].barh(model_names, val_rmses, color=colors, edgecolor="white")
    axes[0, 0].set_title("Validation RMSE by Model")
    axes[0, 0].set_xlabel("RMSE (lower is better)")
    for i, v in enumerate(val_rmses):
        axes[0, 0].text(v + 0.005, i, f"${v*100000:,.0f}", va="center", fontsize=9)

    # Plot 2: Actual vs Predicted
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.3, s=10, color="#3498db")
    axes[0, 1].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        "r--", linewidth=2, label="Perfect",
    )
    axes[0, 1].set_xlabel("Actual Price ($100K)")
    axes[0, 1].set_ylabel("Predicted Price ($100K)")
    axes[0, 1].set_title(f"{best_name}: Actual vs Predicted")
    axes[0, 1].legend()

    # Plot 3: Residual distribution
    residuals = y_test - y_pred_test
    axes[1, 0].hist(residuals, bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 0].set_xlabel("Prediction Error ($100K)")
    axes[1, 0].set_ylabel("Count")

    # Plot 4: Feature importance
    if hasattr(best_pipeline.named_steps.get("model", best_pipeline), "feature_importances_"):
        importances.plot(kind="barh", ax=axes[1, 1], color="#e67e22", edgecolor="white")
        axes[1, 1].set_title("Feature Importances")
        axes[1, 1].set_xlabel("Importance")
    else:
        axes[1, 1].text(0.5, 0.5, "Feature importances\nnot available for\nthis model type",
                        ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig("../notebooks/pipeline_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: pipeline_results.png")
    plt.show()

    # ══════════════════════════════════════════════════════════
    # STEP 8: Save Model for Deployment
    # ══════════════════════════════════════════════════════════
    print_header("STEP 8: Save Model")

    model_dir = Path("../saved_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "best_model_california_housing.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"  Model saved to: {model_path}")
    print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")

    # Demonstrate loading
    loaded_model = joblib.load(model_path)
    sample = X_test.iloc[:3]
    predictions = loaded_model.predict(sample)
    print(f"\n  Verification — predictions from loaded model:")
    for i, (pred, actual) in enumerate(zip(predictions, y_test.iloc[:3])):
        print(f"    Sample {i+1}: Predicted ${pred*100000:,.0f} | "
              f"Actual ${actual*100000:,.0f}")

    print("\n✅ Production Pipeline Complete!")
    print("   The saved model can be loaded in a Flask API or Gradio app.")
    print("   Next: Run 03_bias_audit.py for the ethics practical")


if __name__ == "__main__":
    main()
