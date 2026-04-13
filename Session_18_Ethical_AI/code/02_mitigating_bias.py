import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def main():
    print_header("Mitigating Bias in AI (Removing Proxy Variables)")
    
    # Re-generating the same dataset from script 1
    np.random.seed(42)
    n = 2000
    gender_group = np.random.binomial(1, 0.5, n) 
    experience = np.random.normal(5, 2, n)
    hired_prob = np.clip(0.2 + (0.1 * experience) + (0.4 * gender_group), 0, 1)
    hired_actual = np.random.binomial(1, hired_prob)
    
    df = pd.DataFrame({
        'Experience': experience,
        'Gender_Code': gender_group, 
        'Hired': hired_actual
    })
    
    print("Mitigation Scenario: We ran the audit previously and failed.")
    print("To fix the bias, we will DELETE the 'Gender_Code' column so the AI cannot look at it during training.")
    print("Let's see if this creates a mathematically fair model.\n")
    
    # 1. Mitigation - Drop the Protected Class BEFORE training
    X_fair = df[['Experience']] # Notice 'Gender_Code' is removed!
    y = df['Hired']
    
    fair_model = RandomForestClassifier(max_depth=4, random_state=42)
    fair_model.fit(X_fair, y)
    
    # Generate new predictions
    df['AI_Fair_Prediction'] = fair_model.predict(X_fair)
    
    # 2. Re-auditing the newly trained model!
    print_header("Executing Ethical AI Audit V2")
    
    # Note: Even though the AI didn't *learn* from the Gender code, we STILL 
    # check the protected group against the results to ensure fairness occurred.
    priv_df = df[df['Gender_Code'] == 1]
    priv_approval_rate = priv_df['AI_Fair_Prediction'].mean()
    
    unpriv_df = df[df['Gender_Code'] == 0]
    unpriv_approval_rate = unpriv_df['AI_Fair_Prediction'].mean()
    
    print(f"AI Approval Rate for Privileged Group (1):   {priv_approval_rate * 100:.1f}%")
    print(f"AI Approval Rate for Unprivileged Group (0): {unpriv_approval_rate * 100:.1f}%")
    
    # Calculate Disparate Impact again
    disparate_impact = unpriv_approval_rate / priv_approval_rate
    
    print(f"\n[Mathematical Disparate Impact Ratio]: {disparate_impact:.3f}")
    
    if disparate_impact < 0.8:
        print("❌ FAILED: Ratio is below 0.80.")
    else:
        print("✅ PASSED: Ratio is above 0.80 (The Four-Fifths Rule).")
        print("The AI is now mathematically fair. By removing the explicit demographic column, it can only judge based on pure Experience.")

if __name__ == "__main__":
    main()
