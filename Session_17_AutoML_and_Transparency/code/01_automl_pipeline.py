# Please install TPOT before running:
# pip install tpot scikit-learn pandas

from tpot import TPOTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    print("=" * 60)
    print(" Automated Machine Learning (AutoML) with TPOT ")
    print("=" * 60)
    
    # 1. Load Data
    print("Loading Breast Cancer Dataset...")
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.75, test_size=0.25, random_state=42)
    
    # 2. Configure AutoML
    # Warning: In real projects, generations=100 and population_size=100. 
    # We use 5 and 20 here so the script finishes in 2 minutes instead of 2 hours!
    print("Initializing TPOT Genetic Algorithm Pipeline Search...")
    tpot = TPOTClassifier(
        generations=5,            # How many evolutionary cycles to run
        population_size=20,       # How many pipelines in each generation
        verbosity=2,              # Output progress to terminal
        random_state=42,          # Ensure reproducibility
        n_jobs=-1                 # Use all CPU cores
    )
    
    # 3. Train (AutoML Search)
    print("\n[AutoML Engine Starting] Testing hundreds of Scikit-Learn combinations...\n")
    tpot.fit(X_train, y_train)
    
    # 4. Evaluate Final Perfected Pipeline
    score = tpot.score(X_test, y_test)
    print("\n" + "=" * 60)
    print(f"✅ AutoML Search Complete!")
    print(f"Optimal Pipeline Accuracy on Test Data: {score * 100:.2f}%")
    
    # 5. Export the Code!
    # The true power of TPOT is that it writes the Scikit-Learn code FOR YOU.
    filepath = 'tpot_best_pipeline.py'
    tpot.export(filepath)
    print(f"The exact Python code for the winning pipeline has been saved to: {filepath}")
    print("Open that file to see what combination of Scalers and Algorithms won the evolutionary battle!")

if __name__ == "__main__":
    main()
