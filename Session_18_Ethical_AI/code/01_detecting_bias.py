import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def main():
    print_header("Detecting Bias in AI (Disparate Impact)")
    
    # 1. Generate MOCK BIASED Data
    # Scenario: Tech Hiring. 
    # Let's say Males historically get rejected 30% of the time. 
    # Females historically got rejected 60% of the time.
    np.random.seed(42)
    n = 2000
    
    # 1 = Privileged (Male), 0 = Unprivileged (Female)
    gender_group = np.random.binomial(1, 0.5, n) 
    
    # Years of experience correlates slightly with getting hired
    experience = np.random.normal(5, 2, n)
    
    # Generate the highly biased "Hired" target variable based on historical data
    hired_probability = 0.2 + (0.1 * experience) + (0.4 * gender_group)
    hired_probability = np.clip(hired_probability, 0, 1) # Keep between 0 and 1
    hired_actual = np.random.binomial(1, hired_probability)
    
    df = pd.DataFrame({
        'Experience': experience,
        'Gender_Code': gender_group, # Note: Ethical AI engineers DO look at protected variables during the Audit phase!
        'Hired': hired_actual
    })
    
    # 2. Train the AI Model
    print("Training Random Forest Hiring AI on Historical Data...")
    X = df[['Experience', 'Gender_Code']]
    y = df['Hired']
    
    model = RandomForestClassifier(max_depth=4, random_state=42)
    model.fit(X, y)
    
    # Let's make some predictions!
    df['AI_Prediction'] = model.predict(X)
    acc = accuracy_score(y, df['AI_Prediction'])
    print(f"✅ AI Trained. Overall Accuracy: {acc * 100:.1f}%\n")
    
    # 3. The Ethical Audit (Calculating Disparate Impact)
    print_header("Executing Ethical AI Audit")
    
    # Get Approval Rates for the Privileged Group (Gender = 1)
    priv_df = df[df['Gender_Code'] == 1]
    priv_approval_rate = priv_df['AI_Prediction'].mean()
    
    # Get Approval Rates for the Unprivileged Group (Gender = 0)
    unpriv_df = df[df['Gender_Code'] == 0]
    unpriv_approval_rate = unpriv_df['AI_Prediction'].mean()
    
    print(f"AI Approval Rate for Privileged Group (1):   {priv_approval_rate * 100:.1f}%")
    print(f"AI Approval Rate for Unprivileged Group (0): {unpriv_approval_rate * 100:.1f}%")
    
    # Calculate Disparate Impact
    disparate_impact = unpriv_approval_rate / priv_approval_rate
    
    print(f"\n[Mathematical Disparate Impact Ratio]: {disparate_impact:.3f}")
    
    if disparate_impact < 0.8:
        print("❌ FAILED: Ratio is below 0.80 (The Four-Fifths Rule).")
        print("The AI is heavily biased! It is recommending hiring the privileged group at an unfair rate.")
        print("Do NOT release this to production.")
    else:
        print("✅ PASSED: Ratio is above 0.80. The AI is relatively fair.")

if __name__ == "__main__":
    main()
