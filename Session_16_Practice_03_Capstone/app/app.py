from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)

# ---------------------------------------------------------
# EDA & DATA CLEANING LOGIC (From Sessions 12 & 13)
# ---------------------------------------------------------
def clean_data(df):
    """
    Cleans raw JSON dataframe by removing NaNs and capping massive outliers using IQR.
    """
    # 1. Handle Missing Values
    df = df.dropna(subset=['Spend_USD'])
    
    # 2. Handle Outliers using IQR Cap (Winsorization)
    Q1 = df['Spend_USD'].quantile(0.25)
    Q3 = df['Spend_USD'].quantile(0.75)
    IQR = Q3 - Q1
    
    upper_bound = Q3 + 1.5 * IQR
    # Cap upper outliers (Ignore lower bound as $0 is valid for non-converting users)
    df.loc[df['Spend_USD'] > upper_bound, 'Spend_USD'] = upper_bound
    
    return df

# ---------------------------------------------------------
# CAUSALITY PIPELINE (From Session 15)
# ---------------------------------------------------------
def run_ab_test(df):
    """
    Splits into A/B groups, calculates standard conversion, and runs the T-Test.
    """
    group_a = df[df['Group'] == 'A']['Converted']
    group_b = df[df['Group'] == 'B']['Converted']
    
    cr_a = group_a.mean()
    cr_b = group_b.mean()
    
    # Run T-Test
    t_stat, p_val = stats.ttest_ind(group_b, group_a)
    
    is_significant = bool(p_val < 0.05)
    
    return {
        "Control_Conversion_Rate": round(cr_a * 100, 2),
        "Treatment_Conversion_Rate": round(cr_b * 100, 2),
        "Absolute_Lift": round((cr_b - cr_a) * 100, 2),
        "P_Value": round(p_val, 5),
        "Statistically_Significant": is_significant,
        "Recommendation": "Deploy Group B" if is_significant else "Keep Group A"
    }

# ---------------------------------------------------------
# FLASK API ROUTING (From Session 14)
# ---------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "A/B Testing Engine Online"})

@app.route('/analyze_ab_test', methods=['POST'])
def analyze():
    # 1. Safely grab JSON from the HTTP Request
    try:
        raw_json = request.get_json()
        
        if not raw_json:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        # 2. Convert to Pandas
        df = pd.DataFrame(raw_json)
        
        # 3. Execute EDA Cleaning
        cleaned_df = clean_data(df)
        
        # 4. Execute Causal Analysis
        results = run_ab_test(cleaned_df)
        
        results["status"] = "success"
        results["rows_processed"] = len(cleaned_df)
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == "__main__":
    print("="*60)
    print(" Starting Custom A/B Statistical Microservice...".center(60))
    print("="*60)
    # The default Hugging Face container port is 7860
    app.run(host='0.0.0.0', port=7860, debug=False)
