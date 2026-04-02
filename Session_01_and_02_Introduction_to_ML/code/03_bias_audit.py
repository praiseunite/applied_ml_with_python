"""
============================================================
03_bias_audit.py
Applied Machine Learning Using Python — Session 01
============================================================
Topic: Ethics & Bias Audit on Real-World Data
Level: 🟡 Intermediate → 🔴 Advanced

This script performs a comprehensive bias/fairness audit:
1. Load real-world dataset with known bias issues
2. Train a classification model
3. Measure fairness metrics across protected groups
4. Visualize disparities
5. Implement basic bias mitigation
6. Generate an audit report

Dataset: UCI Adult Census Income Dataset
- 48,842 records of US census data
- Task: Predict if income > $50K/year
- Known bias: Gender and racial disparities
- Reference: Kohavi, R. (1996). "Scaling Up the Accuracy of
  Naive-Bayes Classifiers." KDD.

Usage:
    python 03_bias_audit.py

Requirements:
    pip install numpy pandas scikit-learn matplotlib seaborn
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

plt.style.use("seaborn-v0_8-whitegrid")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def calculate_fairness_metrics(y_true, y_pred, group_labels, group_name):
    """
    Calculate comprehensive fairness metrics for a protected attribute.
    
    Parameters
    ----------
    y_true : array - True labels
    y_pred : array - Predicted labels
    group_labels : array - Group membership labels
    group_name : str - Name of the protected attribute
    
    Returns
    -------
    dict : Fairness metrics per group
    """
    groups = np.unique(group_labels)
    metrics = {}

    print(f"\n  ┌─────────── Fairness Analysis: {group_name} ───────────┐")

    for group in groups:
        mask = group_labels == group
        g_true = y_true[mask]
        g_pred = y_pred[mask]

        # Basic metrics
        accuracy = accuracy_score(g_true, g_pred)
        positive_rate = g_pred.mean()  # Demographic parity metric

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(g_true, g_pred, labels=[0, 1]).ravel()

        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

        metrics[group] = {
            "count": mask.sum(),
            "accuracy": accuracy,
            "positive_rate": positive_rate,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,
            "precision": precision,
        }

        print(f"  │")
        print(f"  │ Group: {group}")
        print(f"  │   Samples:           {mask.sum():>8,}")
        print(f"  │   Accuracy:           {accuracy:>8.2%}")
        print(f"  │   Positive Rate:      {positive_rate:>8.2%}  ← Demographic Parity")
        print(f"  │   True Positive Rate: {tpr:>8.2%}  ← Equalized Odds (TPR)")
        print(f"  │   False Positive Rate:{fpr:>8.2%}  ← Equalized Odds (FPR)")
        print(f"  │   False Negative Rate:{fnr:>8.2%}")
        print(f"  │   Precision:          {precision:>8.2%}  ← Predictive Parity")

    # Calculate disparities
    group_list = list(metrics.keys())
    if len(group_list) >= 2:
        g1, g2 = group_list[0], group_list[1]
        dp_ratio = metrics[g1]["positive_rate"] / max(metrics[g2]["positive_rate"], 1e-10)
        eo_diff = abs(metrics[g1]["tpr"] - metrics[g2]["tpr"])
        fpr_diff = abs(metrics[g1]["fpr"] - metrics[g2]["fpr"])

        print(f"  │")
        print(f"  ├─── Disparity Metrics ───")
        print(f"  │")
        print(f"  │ Demographic Parity Ratio: {dp_ratio:.3f}")
        print(f"  │   (Should be between 0.8 and 1.25 — '80% rule')")
        dp_status = "✅ PASS" if 0.8 <= dp_ratio <= 1.25 else "❌ FAIL"
        print(f"  │   Status: {dp_status}")
        print(f"  │")
        print(f"  │ Equalized Odds TPR Difference: {eo_diff:.3f}")
        print(f"  │   (Should be < 0.1 for fairness)")
        eo_status = "✅ PASS" if eo_diff < 0.1 else "❌ FAIL"
        print(f"  │   Status: {eo_status}")
        print(f"  │")
        print(f"  │ FPR Difference: {fpr_diff:.3f}")
        fpr_status = "✅ PASS" if fpr_diff < 0.05 else "❌ FAIL"
        print(f"  │   Status: {fpr_status}")

    print(f"  └{'─' * 50}┘")
    return metrics


def main():
    # ══════════════════════════════════════════════════════════
    # STEP 1: Load the Adult Census Dataset
    # ══════════════════════════════════════════════════════════
    print_header("STEP 1: Loading UCI Adult Census Dataset")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]

    print("  Downloading dataset from UCI ML Repository...")
    try:
        df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    except Exception:
        # Fallback: try with local file or alternative URL
        print("  ⚠️  Download failed. Trying alternative source...")
        alt_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
        df = pd.read_csv(alt_url, names=columns, na_values="?", skipinitialspace=True)

    df = df.dropna()
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Explore the Data for Potential Bias
    # ══════════════════════════════════════════════════════════
    print_header("STEP 2: Exploring Data for Bias Indicators")

    # Income distribution
    print(f"\n  Income Distribution:")
    income_counts = df["income"].value_counts()
    for income, count in income_counts.items():
        print(f"    {income:>5s}: {count:>6,} ({count/len(df):.1%})")

    # Gender distribution
    print(f"\n  Gender Distribution:")
    for gender in df["sex"].unique():
        count = (df["sex"] == gender).sum()
        high_income = ((df["sex"] == gender) & (df["income"] == ">50K")).mean()
        print(f"    {gender:>8s}: {count:>6,} samples | "
              f"High income rate: {high_income:.1%}")

    # Race distribution
    print(f"\n  Race Distribution:")
    for race in df["race"].unique():
        count = (df["race"] == race).sum()
        high_income = ((df["race"] == race) & (df["income"] == ">50K")).mean()
        print(f"    {race:>20s}: {count:>6,} samples | "
              f"High income rate: {high_income:.1%}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Prepare Data and Train Model
    # ══════════════════════════════════════════════════════════
    print_header("STEP 3: Training Classification Model")

    # Save original protected attributes before encoding
    df_original = df.copy()

    # Encode categorical variables
    le_dict = {}
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le

    # Prepare features and target
    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Keep track of protected attributes in test set
    test_indices = X_test.index
    test_sex = df_original.loc[test_indices, "sex"].values
    test_race = df_original.loc[test_indices, "race"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Overall performance
    print(f"\n  Overall Model Performance:")
    print(f"  {'─' * 40}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
    print(f"  Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.2%}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.2%}")

    # ══════════════════════════════════════════════════════════
    # STEP 4: Bias Audit — Gender
    # ══════════════════════════════════════════════════════════
    print_header("STEP 4: Bias Audit — Gender")
    gender_metrics = calculate_fairness_metrics(
        y_test.values, y_pred, test_sex, "Gender"
    )

    # ══════════════════════════════════════════════════════════
    # STEP 5: Bias Audit — Race
    # ══════════════════════════════════════════════════════════
    print_header("STEP 5: Bias Audit — Race")
    race_metrics = calculate_fairness_metrics(
        y_test.values, y_pred, test_race, "Race"
    )

    # ══════════════════════════════════════════════════════════
    # STEP 6: Feature Importance (Transparency Check)
    # ══════════════════════════════════════════════════════════
    print_header("STEP 6: Feature Importance — Transparency")

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    print(f"\n  Top 10 Features Driving Predictions:")
    print(f"  {'─' * 50}")
    for rank, (feat, imp) in enumerate(importances.head(10).items(), 1):
        protected = ""
        if feat in ["sex", "race", "native_country"]:
            protected = "  ⚠️  PROTECTED ATTRIBUTE"
        bar = "█" * int(imp * 200)
        print(f"  {rank:2d}. {feat:20s} {imp:.4f} {bar}{protected}")

    if "sex" in importances.head(5).index or "race" in importances.head(5).index:
        print(f"\n  ❌ WARNING: Protected attributes are among the top features!")
        print(f"     The model may be directly using protected characteristics")
        print(f"     to make predictions — this is a significant fairness concern.")
    else:
        print(f"\n  ✅ Protected attributes are not in the top 5 features.")
        print(f"     However, proxy variables (e.g., occupation, education) may")
        print(f"     still encode protected information indirectly.")

    # ══════════════════════════════════════════════════════════
    # STEP 7: Visualizations
    # ══════════════════════════════════════════════════════════
    print_header("STEP 7: Bias Audit Visualizations")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("AI Bias Audit — UCI Adult Census Income", fontsize=16, fontweight="bold")

    # Plot 1: Positive prediction rate by gender (Demographic Parity)
    genders = list(gender_metrics.keys())
    pos_rates = [gender_metrics[g]["positive_rate"] for g in genders]
    colors = ["#3498db", "#e74c3c"]
    bars = axes[0, 0].bar(genders, pos_rates, color=colors, edgecolor="white", width=0.5)
    axes[0, 0].set_title("Demographic Parity: Positive Rate by Gender")
    axes[0, 0].set_ylabel("P(Predicted High Income)")
    axes[0, 0].axhline(y=np.mean(pos_rates), color="gray", linestyle="--",
                        label=f"Average: {np.mean(pos_rates):.2%}")
    axes[0, 0].legend()
    for bar, rate in zip(bars, pos_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")

    # Plot 2: TPR and FPR by gender (Equalized Odds)
    x = np.arange(len(genders))
    width = 0.3
    tprs = [gender_metrics[g]["tpr"] for g in genders]
    fprs = [gender_metrics[g]["fpr"] for g in genders]
    axes[0, 1].bar(x - width / 2, tprs, width, label="TPR", color="#2ecc71")
    axes[0, 1].bar(x + width / 2, fprs, width, label="FPR", color="#e74c3c")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(genders)
    axes[0, 1].set_title("Equalized Odds: TPR & FPR by Gender")
    axes[0, 1].legend()

    # Plot 3: Positive prediction rate by race
    races = list(race_metrics.keys())
    race_pos_rates = [race_metrics[r]["positive_rate"] for r in races]
    race_colors = plt.cm.Set2(np.linspace(0, 1, len(races)))
    bars = axes[1, 0].barh(races, race_pos_rates, color=race_colors, edgecolor="white")
    axes[1, 0].set_title("Demographic Parity: Positive Rate by Race")
    axes[1, 0].set_xlabel("P(Predicted High Income)")
    for bar, rate in zip(bars, race_pos_rates):
        axes[1, 0].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{rate:.1%}", va="center", fontsize=10)

    # Plot 4: Feature importance
    top_features = importances.head(10)
    colors_fi = ["#e74c3c" if f in ["sex", "race", "native_country"]
                 else "#3498db" for f in top_features.index]
    top_features.plot(kind="barh", ax=axes[1, 1], color=colors_fi, edgecolor="white")
    axes[1, 1].set_title("Feature Importance (Red = Protected)")
    axes[1, 1].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("../notebooks/bias_audit_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: bias_audit_results.png")
    plt.show()

    # ══════════════════════════════════════════════════════════
    # STEP 8: Bias Mitigation — Removing Protected Features
    # ══════════════════════════════════════════════════════════
    print_header("STEP 8: Bias Mitigation — Removing Protected Features")

    # Strategy: Remove protected attributes and retrain
    protected_cols = ["sex", "race", "native_country"]
    X_fair = X.drop(columns=protected_cols, errors="ignore")

    X_train_fair, X_test_fair, y_train_fair, y_test_fair = train_test_split(
        X_fair, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler_fair = StandardScaler()
    X_train_fair_scaled = scaler_fair.fit_transform(X_train_fair)
    X_test_fair_scaled = scaler_fair.transform(X_test_fair)

    model_fair = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    model_fair.fit(X_train_fair_scaled, y_train_fair)
    y_pred_fair = model_fair.predict(X_test_fair_scaled)

    # Compare results
    print(f"\n  Model Performance Comparison:")
    print(f"  {'Metric':<20s} {'Original':>12s} {'Fair Model':>12s} {'Change':>10s}")
    print(f"  {'─' * 55}")

    orig_acc = accuracy_score(y_test, y_pred)
    fair_acc = accuracy_score(y_test_fair, y_pred_fair)
    orig_f1 = f1_score(y_test, y_pred)
    fair_f1 = f1_score(y_test_fair, y_pred_fair)

    print(f"  {'Accuracy':<20s} {orig_acc:>12.2%} {fair_acc:>12.2%} "
          f"{(fair_acc - orig_acc):>+10.2%}")
    print(f"  {'F1 Score':<20s} {orig_f1:>12.2%} {fair_f1:>12.2%} "
          f"{(fair_f1 - orig_f1):>+10.2%}")

    # Recalculate gender fairness on fair model
    test_sex_fair = df_original.loc[X_test_fair.index, "sex"].values
    print(f"\n  Gender Fairness After Mitigation:")
    gender_metrics_fair = calculate_fairness_metrics(
        y_test_fair.values, y_pred_fair, test_sex_fair, "Gender (Fair Model)"
    )

    # ══════════════════════════════════════════════════════════
    # STEP 9: Generate Audit Report Summary
    # ══════════════════════════════════════════════════════════
    print_header("AUDIT REPORT SUMMARY")

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │            RESPONSIBLE AI AUDIT REPORT                  │
  │            UCI Adult Census Income Model                │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  Dataset:     UCI Adult Census (30,162 records)          │
  │  Task:        Income classification (>$50K / ≤$50K)     │
  │  Model:       Random Forest (200 trees, max_depth=15)   │
  │  Accuracy:    {orig_acc:.2%}                                 │
  │                                                         │
  │  FINDINGS:                                              │
  │                                                         │
  │  1. GENDER BIAS:                                        │
  │     • Male positive prediction rate significantly       │
  │       higher than female                                │
  │     • True positive rate gap indicates the model        │
  │       is better at identifying high-income males        │
  │                                                         │
  │  2. RACIAL DISPARITIES:                                 │
  │     • Significant variation in positive prediction      │
  │       rates across racial groups                        │
  │     • Some groups have very few samples, making         │
  │       metrics unreliable                                │
  │                                                         │
  │  3. PROTECTED FEATURES:                                 │
  │     • Model uses sex and relationship status            │
  │       as significant predictors                         │
  │     • Even after removing protected features,           │
  │       proxy variables may encode bias                   │
  │                                                         │
  │  RECOMMENDATION:                                        │
  │     This model should NOT be deployed for real          │
  │     lending/hiring decisions without additional         │
  │     bias mitigation and human oversight.                │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
    """)

    print("✅ Bias Audit Complete!")
    print("   📊 See bias_audit_results.png for visualizations")
    print("   📝 Use this analysis as the basis for your Portfolio component:")
    print("      'Responsible AI Audit Report'")
    print()
    print("   Key Takeaways:")
    print("   1. ALWAYS audit ML models for bias before deployment")
    print("   2. Removing protected features ≠ fair model (proxy bias)")
    print("   3. Fairness is a DESIGN CHOICE, not just a technical metric")
    print("   4. Different stakeholders may prioritize different fairness metrics")


if __name__ == "__main__":
    main()
