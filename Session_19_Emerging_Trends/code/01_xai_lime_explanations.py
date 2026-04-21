# Session 19 — Script 01: Explainable AI with LIME
# =================================================
# This script demonstrates how to use LIME (Local Interpretable Model-agnostic
# Explanations) to understand exactly WHY a Black-Box AI model made a specific
# prediction for a specific patient.
#
# Real-World Scenario:
# You are a Data Scientist at a hospital. The AI flags Patient #42 as "High Risk
# for Heart Disease." The attending cardiologist refuses to act on it unless you
# explain WHICH medical factors triggered the prediction and by how much.
#
# Dependencies: pip install lime scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70 + "\n")

def generate_heart_disease_data(n=2000):
    """
    Generate a realistic mock heart disease dataset.
    Features are modeled after the UCI Heart Disease dataset.
    """
    np.random.seed(42)
    
    # Patient demographics and vitals
    age = np.random.normal(55, 12, n).clip(25, 85).astype(int)
    sex = np.random.binomial(1, 0.6, n)  # 1=Male, 0=Female
    
    # Clinical measurements
    resting_bp = np.random.normal(130, 18, n).clip(90, 200).astype(int)
    cholesterol = np.random.normal(240, 45, n).clip(120, 400).astype(int)
    fasting_blood_sugar = (np.random.normal(110, 30, n) > 120).astype(int)
    max_heart_rate = (220 - age - np.random.normal(0, 15, n)).clip(70, 210).astype(int)
    
    # Exercise-induced angina (chest pain during exercise)
    exercise_angina = np.random.binomial(1, 0.3, n)
    
    # ST depression (ECG measurement — higher = worse)
    st_depression = np.random.exponential(1.0, n).clip(0, 6).round(1)
    
    # Number of major vessels colored by fluoroscopy (0-3)
    num_vessels = np.random.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.13, 0.07])
    
    # Generate risk score to create target variable
    risk_score = (
        0.03 * age
        + 0.15 * sex
        + 0.01 * resting_bp
        + 0.005 * cholesterol
        - 0.02 * max_heart_rate
        + 0.8 * exercise_angina
        + 0.5 * st_depression
        + 0.7 * num_vessels
        + 0.3 * fasting_blood_sugar
        + np.random.normal(0, 0.5, n)  # noise
    )
    
    # Normalize to probability and binarize
    risk_prob = 1 / (1 + np.exp(-(risk_score - np.median(risk_score))))
    heart_disease = np.random.binomial(1, risk_prob)
    
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBloodSugar': fasting_blood_sugar,
        'MaxHeartRate': max_heart_rate,
        'ExerciseAngina': exercise_angina,
        'ST_Depression': st_depression,
        'NumVessels': num_vessels,
        'HeartDisease': heart_disease
    })
    return df

def main():
    print_header("Explainable AI (XAI) with LIME")
    
    # ─── Stage 1: Generate and Explore Data ──────────────────────────────
    print("📊 Generating synthetic Heart Disease dataset (2,000 patients)...\n")
    df = generate_heart_disease_data(n=2000)
    
    feature_names = [c for c in df.columns if c != 'HeartDisease']
    X = df[feature_names].values
    y = df['HeartDisease'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    print(f"   Features: {feature_names}")
    print(f"   Class balance: {np.mean(y):.1%} positive (Heart Disease)\n")
    
    # ─── Stage 2: Train a Complex Black-Box Model ────────────────────────
    print_header("Training Black-Box Model (Gradient Boosting)")
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model Trained Successfully.")
    print(f"   Test Accuracy: {acc * 100:.1f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    print("⚠️  The model is highly accurate — but it has 200 trees with depth 5.")
    print("   No human can read the math. This is the BLACK-BOX problem.\n")
    
    # ─── Stage 3: LIME Explanation for a Single Patient ──────────────────
    print_header("LIME: Explaining Patient #42's Prediction")
    
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("❌ LIME is not installed. Install it with:")
        print("   pip install lime")
        print("\nFalling back to manual feature importance analysis...\n")
        _fallback_explanation(model, X_test, feature_names, patient_idx=42)
        return
    
    # Create the LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Low Risk', 'High Risk'],
        mode='classification',
        random_state=42
    )
    
    # Select a specific patient to explain
    patient_idx = 42
    patient_data = X_test[patient_idx]
    patient_prediction = model.predict([patient_data])[0]
    patient_proba = model.predict_proba([patient_data])[0]
    
    print(f"Patient #{patient_idx} Data:")
    for fname, val in zip(feature_names, patient_data):
        print(f"   {fname:>20s}: {val}")
    
    print(f"\n🤖 AI Prediction: {'🔴 HIGH RISK' if patient_prediction == 1 else '🟢 LOW RISK'}")
    print(f"   Confidence: Low Risk = {patient_proba[0]:.1%}, High Risk = {patient_proba[1]:.1%}")
    
    # Generate the LIME explanation
    print("\n🔬 Running LIME analysis (generating 5,000 perturbations)...")
    explanation = explainer.explain_instance(
        patient_data,
        model.predict_proba,
        num_features=len(feature_names),
        num_samples=5000
    )
    
    # Display the explanation
    print_header("LIME Explanation Results")
    print("Each line below shows a feature condition and its CONTRIBUTION to the prediction.")
    print("Positive values PUSH toward High Risk. Negative values PUSH toward Low Risk.\n")
    
    explanation_list = explanation.as_list()
    
    print(f"{'Feature Condition':<45} {'Contribution':>12}")
    print("-" * 60)
    
    for feature_condition, weight in sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True):
        direction = "→ HIGH RISK" if weight > 0 else "→ LOW RISK "
        bar_length = int(abs(weight) * 50)
        bar = "█" * min(bar_length, 30)
        print(f"  {feature_condition:<43} {weight:>+.4f}  {direction} {bar}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION FOR THE DOCTOR:")
    print("=" * 70)
    
    # Sort by absolute contribution
    top_factors = sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True)[:3]
    
    if patient_prediction == 1:
        print(f"\nPatient #{patient_idx} was classified as HIGH RISK primarily because:")
    else:
        print(f"\nPatient #{patient_idx} was classified as LOW RISK primarily because:")
    
    for i, (condition, weight) in enumerate(top_factors, 1):
        direction = "increased" if weight > 0 else "decreased"
        print(f"  {i}. '{condition}' {direction} the risk score by {abs(weight):.4f}")
    
    print(f"\n✅ The cardiologist now knows EXACTLY why the AI flagged this patient.")
    print(f"   They can verify whether these reasons are clinically sound before acting.\n")

def _fallback_explanation(model, X_test, feature_names, patient_idx=42):
    """Fallback when LIME is not installed — uses built-in feature importances."""
    print_header("Fallback: Global Feature Importance (Built-in)")
    print("Without LIME, we can only show GLOBAL feature importance,")
    print("not per-patient explanations. Install LIME for the full experience.\n")
    
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print(f"{'Feature':<25} {'Importance':>12}")
    print("-" * 40)
    for idx in sorted_idx:
        bar = "█" * int(importances[idx] * 50)
        print(f"  {feature_names[idx]:<23} {importances[idx]:>10.4f}  {bar}")
    
    patient_data = X_test[patient_idx]
    prediction = model.predict([patient_data])[0]
    proba = model.predict_proba([patient_data])[0]
    
    print(f"\nPatient #{patient_idx} Prediction: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
    print(f"Confidence: {max(proba) * 100:.1f}%")
    print("\n⚠️  To see WHY this specific patient was flagged, install LIME:")
    print("   pip install lime\n")

if __name__ == "__main__":
    main()
