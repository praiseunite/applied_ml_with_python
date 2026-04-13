import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import os

def main():
    print("="*50)
    print(" Step 1: Model Serialization ")
    print("="*50)
    
    # 1. Generate a mock synthetic dataset
    print("\nTraining a Random Forest Model...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=4, 
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    
    feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
    df_X = pd.DataFrame(X, columns=feature_names)
    
    # 2. Train the model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(df_X, y)
    print("✅ Model Training Complete.")
    
    # 3. Serialize (Save to Disk)
    model_path = 'my_production_model.joblib'
    joblib.dump(model, model_path)
    
    # Check file size
    file_size_kb = os.path.getsize(model_path) / 1024
    
    print(f"\n✅ Model Successfully Serialized!")
    print(f"File Saved As : {model_path}")
    print(f"File Size     : {file_size_kb:.1f} KB")
    
    print("\nNow that the 'intelligence' is saved to the hard drive,")
    print("you can run `02_flask_api_server.py` to start the web server!")

if __name__ == "__main__":
    main()
