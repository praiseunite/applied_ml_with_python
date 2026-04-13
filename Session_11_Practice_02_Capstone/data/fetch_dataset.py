import os
import pandas as pd

def fetch_predictive_maintenance_data():
    """
    Downloads the AI4I 2020 Predictive Maintenance Dataset directly from the UCI ML Repository.
    No API keys are required. This ensures every student gets the exact same dataset flawlessly.
    """
    print("=" * 60)
    print(" 📡 Fetching Industrial IoT Dataset...".center(60))
    print("=" * 60)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    
    try:
        print("\n[1/3] Downloading from UCI ML Repository...")
        df = pd.read_csv(url)
        print(f"      ✅ Successfully downloaded {len(df)} rows of data.")
        
        print("\n[2/3] Performing basic formatting...")
        # Drop irrelevant identifiers to prevent data leakage
        if 'UDI' in df.columns:
            df = df.drop(columns=['UDI'])
        if 'Product ID' in df.columns:
            df = df.drop(columns=['Product ID'])
            
        # The target variable is 'Machine failure', but it also provides specific failure types (TWF, HDF, PWF, OSF, RNF).
        # We will keep 'Machine failure' as our primary binary label.
        print("      ✅ Formatting complete.")
        
        print("\n[3/3] Saving data locally...")
        output_file = "predictive_maintenance.csv"
        df.to_csv(output_file, index=False)
        print(f"      ✅ Dataset saved to: {os.path.abspath(output_file)}")
        
        # Give students a quick overview of the imbalance
        failures = df['Machine failure'].sum()
        total = len(df)
        print("\n📊 Dataset Imbalance Overview:")
        print(f"   Normal Operations: {total - failures} ({(total-failures)/total*100:.1f}%)")
        print(f"   Machine Failures:   {failures} ({(failures)/total*100:.1f}%)")
        print("   ⚠️ Notice the severe imbalance! You will need to handle this in your notebook.")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("Please check your internet connection or ensure you have pandas installed.")
        
if __name__ == "__main__":
    fetch_predictive_maintenance_data()
