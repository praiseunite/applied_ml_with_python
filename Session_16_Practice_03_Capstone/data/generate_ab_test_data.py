import pandas as pd
import numpy as np
import json
import os

def create_raw_ab_data():
    """
    Generates a synthetic, highly messy dataset of user purchase histories representing an A/B test.
    Includes NaNs and massive outliers so students must use EDA skills to clean it.
    """
    print("=" * 60)
    print(" 🛒 Generating Raw E-Commerce A/B Data...".center(60))
    print("=" * 60)
    
    np.random.seed(42)
    n_users = 2500 # Per group
    
    # 1. Base Logic
    # Group A (Blue Button): 8% conversion, avg spend $50
    # Group B (Green Button): 11.5% conversion, avg spend $55
    group_a_conv = np.random.binomial(1, 0.08, n_users)
    group_a_spend = np.where(group_a_conv == 1, np.random.normal(50, 15, n_users), 0)
    
    group_b_conv = np.random.binomial(1, 0.115, n_users)
    group_b_spend = np.where(group_b_conv == 1, np.random.normal(55, 18, n_users), 0)
    
    df_a = pd.DataFrame({'User_ID': range(1, n_users+1), 'Group': 'A', 'Converted': group_a_conv, 'Spend_USD': group_a_spend})
    df_b = pd.DataFrame({'User_ID': range(n_users+1, (n_users*2)+1), 'Group': 'B', 'Converted': group_b_conv, 'Spend_USD': group_b_spend})
    
    df = pd.concat([df_a, df_b]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Introduce EDA Challenges!
    print("💉 Injecting missing values (NaNs)...")
    nan_indices = np.random.choice(df.index, size=150, replace=False)
    df.loc[nan_indices, 'Spend_USD'] = np.nan
    
    print("😈 Injecting severe outliers (Whale Spenders)...")
    outlier_indices_a = np.random.choice(df[df['Group']=='A'].index, size=5, replace=False)
    outlier_indices_b = np.random.choice(df[df['Group']=='B'].index, size=5, replace=False)
    df.loc[outlier_indices_a, 'Spend_USD'] = np.random.uniform(25000, 50000, 5) # Group A whales
    df.loc[outlier_indices_b, 'Spend_USD'] = np.random.uniform(25000, 50000, 5) # Group B whales
    
    # 3. Save as JSON (To mimic an API payload)
    output_file = "raw_ab_data.json"
    
    # Exporting as records format is ideal for HTTP JSON
    data_dict = df.to_dict(orient='records')
    
    with open(output_file, 'w') as f:
        json.dump(data_dict, f, indent=4)
        
    print(f"\n✅ Successfully generated {len(df)} rows.")
    print(f"✅ Saved to: {os.path.abspath(output_file)}")
    print("  -> Warning: Data contains missing values and extreme anomalies.")
    print("  -> Do NOT run the T-Test until you clean the data using IQR/Z-scores!")

if __name__ == "__main__":
    create_raw_ab_data()
