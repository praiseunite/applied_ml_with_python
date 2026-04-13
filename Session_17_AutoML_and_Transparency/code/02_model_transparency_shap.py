# Please install SHAP before running:
# pip install shap scikit-learn matplotlib

import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print(" Model Transparency with SHAP ")
    print("=" * 60)
    
    # 1. Load Data
    cancer = load_breast_cancer()
    feature_names = cancer.feature_names
    X = pd.DataFrame(cancer.data, columns=feature_names)
    y = cancer.target # 0 = Malignant, 1 = Benign
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train a "Black-Box" Model
    # Random Forests consist of 100+ deep decision trees. 
    # It is impossible for a human to mathematically read the model's raw logic.
    print("Training Black-box Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model Trained.\n")
    
    # 3. Explaining the Model with SHAP
    print("Analyzing Mathematical feature contributions using SHAP...")
    
    # Create the SHAP Explainer object
    explainer = shap.TreeExplainer(model)
    
    # Calculate exact SHAP values for every single patient in the Test Set
    shap_values = explainer.shap_values(X_test)
    
    # 4. Make a Single Prediction Transparent
    patient_idx = 0 
    prediction = model.predict(X_test.iloc[[patient_idx]])[0]
    result_text = "Benign" if prediction == 1 else "Malignant"
    
    print(f"\n[Transparency Report - Patient #{patient_idx}]")
    print(f"The Black-Box Model predicts this tumor is: {result_text}")
    print("Why? Generating Explainer Graph...")
    
    # Generate the Summary Plot (This shows global feature importance)
    print("Saving Global Summary Plot to 'shap_summary_plot.png'...")
    
    plt.figure(figsize=(10, 6))
    
    # Depending on SHAP version, shap_values might be a list (for classification).
    # Index 1 explains the logic for predicting purely "Class 1" (Benign)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
        
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=200)
    print("✅ Complete! Open `shap_summary_plot.png` to see exactly which features drove the black-box reasoning.")

if __name__ == "__main__":
    main()
