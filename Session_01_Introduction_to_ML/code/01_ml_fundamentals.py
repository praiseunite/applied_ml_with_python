"""
============================================================
01_ml_fundamentals.py
Applied Machine Learning Using Python — Session 01
============================================================
Topic: Complete ML Pipeline from Scratch
Level: 🟢 Beginner → 🟡 Intermediate

This script demonstrates a complete Machine Learning pipeline:
1. Problem Definition
2. Data Collection & Exploration
3. Data Preprocessing
4. Model Training (Multiple Algorithms)
5. Model Evaluation & Comparison
6. Visualization of Results

Dataset: Scikit-learn Diabetes Dataset (Real Medical Data)
- 442 patients, 10 baseline features
- Target: Disease progression one year after baseline
- Reference: Bradley Efron et al. (2004), "Least Angle Regression"

Usage:
    python 01_ml_fundamentals.py

Requirements:
    pip install numpy pandas scikit-learn matplotlib seaborn
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Set style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def main():
    # ══════════════════════════════════════════════════════════
    # STEP 1: Problem Definition
    # ══════════════════════════════════════════════════════════
    print_header("STEP 1: Problem Definition")
    print("""
    OBJECTIVE: Predict diabetes disease progression one year
    after baseline measurements.
    
    TYPE: Regression (predicting a continuous value)
    
    METRIC: RMSE (Root Mean Squared Error) — lower is better
            R² Score — closer to 1.0 is better
    
    BUSINESS VALUE: Help doctors identify high-risk patients
    early so they can receive preventive care.
    """)

    # ══════════════════════════════════════════════════════════
    # STEP 2: Data Collection & Exploration
    # ══════════════════════════════════════════════════════════
    print_header("STEP 2: Data Collection & Exploration")

    # Load the dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="disease_progression")

    print(f"Dataset Shape: {X.shape}")
    print(f"Number of Patients: {X.shape[0]}")
    print(f"Number of Features: {X.shape[1]}")
    print(f"\nFeature Descriptions:")
    feature_descriptions = {
        "age": "Age of patient",
        "sex": "Sex of patient",
        "bmi": "Body Mass Index",
        "bp": "Average blood pressure",
        "s1": "Total serum cholesterol (tc)",
        "s2": "Low-density lipoproteins (ldl)",
        "s3": "High-density lipoproteins (hdl)",
        "s4": "Total cholesterol / HDL (tch)",
        "s5": "Log of serum triglycerides (ltg)",
        "s6": "Blood sugar level (glu)",
    }
    for feat, desc in feature_descriptions.items():
        print(f"  {feat:6s} → {desc}")

    print(f"\nTarget Variable: Disease progression (quantitative measure)")
    print(f"  Range: {y.min():.0f} to {y.max():.0f}")
    print(f"  Mean:  {y.mean():.1f}")
    print(f"  Std:   {y.std():.1f}")

    # Statistical summary
    print(f"\nStatistical Summary:")
    print(X.describe().round(3).to_string())

    # Check for missing values
    missing = X.isnull().sum().sum()
    print(f"\nMissing Values: {missing} (clean dataset)")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Data Visualization (EDA)
    # ══════════════════════════════════════════════════════════
    print_header("STEP 3: Data Visualization")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Exploratory Data Analysis — Diabetes Dataset", fontsize=16, fontweight="bold")

    # Plot 1: Target distribution
    axes[0, 0].hist(y, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0, 0].set_title("Target Distribution")
    axes[0, 0].set_xlabel("Disease Progression")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(y.mean(), color="red", linestyle="--", label=f"Mean: {y.mean():.0f}")
    axes[0, 0].legend()

    # Plot 2: BMI vs Target
    axes[0, 1].scatter(X["bmi"], y, alpha=0.4, color="#2ecc71", s=20)
    axes[0, 1].set_title("BMI vs Disease Progression")
    axes[0, 1].set_xlabel("BMI (normalized)")
    axes[0, 1].set_ylabel("Disease Progression")

    # Plot 3: Blood Pressure vs Target
    axes[0, 2].scatter(X["bp"], y, alpha=0.4, color="#e74c3c", s=20)
    axes[0, 2].set_title("Blood Pressure vs Disease Progression")
    axes[0, 2].set_xlabel("Blood Pressure (normalized)")
    axes[0, 2].set_ylabel("Disease Progression")

    # Plot 4: Correlation heatmap
    corr_with_target = X.corrwith(y).sort_values(ascending=False)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in corr_with_target.values]
    axes[1, 0].barh(corr_with_target.index, corr_with_target.values, color=colors)
    axes[1, 0].set_title("Feature Correlation with Target")
    axes[1, 0].set_xlabel("Correlation")
    axes[1, 0].axvline(0, color="black", linewidth=0.5)

    # Plot 5: Feature correlations
    top_features = corr_with_target.head(3).index.tolist()
    for i, feat in enumerate(top_features):
        axes[1, 1].scatter(X[feat], y, alpha=0.3, s=15, label=feat)
    axes[1, 1].set_title("Top 3 Correlated Features")
    axes[1, 1].set_xlabel("Feature Value")
    axes[1, 1].set_ylabel("Disease Progression")
    axes[1, 1].legend()

    # Plot 6: Box plot of features
    axes[1, 2].boxplot(X.values, labels=X.columns, vert=True)
    axes[1, 2].set_title("Feature Distributions")
    axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("../notebooks/eda_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: eda_results.png")
    plt.show()

    # ══════════════════════════════════════════════════════════
    # STEP 4: Data Preprocessing
    # ══════════════════════════════════════════════════════════
    print_header("STEP 4: Data Preprocessing")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.0%})")
    print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.0%})")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nAfter scaling (first feature):")
    print(f"  Training mean: {X_train_scaled[:, 0].mean():.6f} (should be ~0)")
    print(f"  Training std:  {X_train_scaled[:, 0].std():.6f} (should be ~1)")

    # ══════════════════════════════════════════════════════════
    # STEP 5: Model Training & Comparison
    # ══════════════════════════════════════════════════════════
    print_header("STEP 5: Model Training & Comparison")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "SVR": SVR(kernel="rbf", C=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }

    results = []

    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        # Cross-validation (5-fold)
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5,
            scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(-cv_scores).std()

        results.append({
            "Model": name,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Test MAE": test_mae,
            "Test R²": test_r2,
            "CV RMSE": cv_rmse,
            "CV Std": cv_std,
        })

        status = "✅" if test_r2 > 0.3 else "⚠️"
        print(f"  {status} {name:25s} | Test RMSE: {test_rmse:7.2f} | "
              f"R²: {test_r2:.3f} | CV RMSE: {cv_rmse:.2f} ± {cv_std:.2f}")

    results_df = pd.DataFrame(results).sort_values("Test RMSE")

    # ══════════════════════════════════════════════════════════
    # STEP 6: Results Visualization
    # ══════════════════════════════════════════════════════════
    print_header("STEP 6: Results Visualization")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Comparison — Diabetes Prediction", fontsize=16, fontweight="bold")

    # Plot 1: RMSE Comparison
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results_df)))
    axes[0, 0].barh(
        results_df["Model"], results_df["Test RMSE"],
        color=colors, edgecolor="white"
    )
    axes[0, 0].set_title("Test RMSE (Lower is Better)")
    axes[0, 0].set_xlabel("RMSE")

    # Plot 2: R² Comparison
    r2_colors = plt.cm.RdYlGn(
        (results_df["Test R²"].values - results_df["Test R²"].min()) /
        (results_df["Test R²"].max() - results_df["Test R²"].min() + 1e-10)
    )
    axes[0, 1].barh(
        results_df["Model"], results_df["Test R²"],
        color=r2_colors, edgecolor="white"
    )
    axes[0, 1].set_title("R² Score (Higher is Better)")
    axes[0, 1].set_xlabel("R²")

    # Plot 3: Train vs Test RMSE (detect overfitting)
    x_pos = range(len(results_df))
    width = 0.35
    axes[1, 0].bar(
        [p - width / 2 for p in x_pos], results_df["Train RMSE"],
        width, label="Train", color="#3498db", alpha=0.8
    )
    axes[1, 0].bar(
        [p + width / 2 for p in x_pos], results_df["Test RMSE"],
        width, label="Test", color="#e74c3c", alpha=0.8
    )
    axes[1, 0].set_xticks(list(x_pos))
    axes[1, 0].set_xticklabels(results_df["Model"], rotation=45, ha="right")
    axes[1, 0].set_title("Train vs Test RMSE (Overfitting Check)")
    axes[1, 0].legend()
    axes[1, 0].set_ylabel("RMSE")

    # Plot 4: Actual vs Predicted (best model)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_scaled)

    axes[1, 1].scatter(y_test, y_pred_best, alpha=0.5, color="#3498db", s=40)
    axes[1, 1].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        "r--", linewidth=2, label="Perfect Prediction"
    )
    axes[1, 1].set_xlabel("Actual Disease Progression")
    axes[1, 1].set_ylabel("Predicted Disease Progression")
    axes[1, 1].set_title(f"Best Model: {best_model_name}")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("../notebooks/model_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved: model_comparison.png")
    plt.show()

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print_header("FINAL RESULTS")
    print(f"\n{'Model':25s} {'Test RMSE':>10s} {'Test R²':>10s} {'CV RMSE':>10s}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        best = " ← BEST" if row["Model"] == best_model_name else ""
        print(f"{row['Model']:25s} {row['Test RMSE']:10.2f} "
              f"{row['Test R²']:10.3f} {row['CV RMSE']:10.2f}{best}")

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test RMSE: {results_df.iloc[0]['Test RMSE']:.2f}")
    print(f"   Test R²:   {results_df.iloc[0]['Test R²']:.3f}")
    print(f"   CV RMSE:   {results_df.iloc[0]['CV RMSE']:.2f} ± {results_df.iloc[0]['CV Std']:.2f}")

    print("\n✅ Session 1 Pipeline Complete!")
    print("   Next: Run 02_sklearn_pipeline.py for the production-grade version")


if __name__ == "__main__":
    main()
