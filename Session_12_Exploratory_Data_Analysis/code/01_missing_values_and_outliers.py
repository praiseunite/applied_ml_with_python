import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def create_messy_data():
    """Generates a synthetic DataFrame with missing values and extreme outliers."""
    np.random.seed(42)
    data = {
        'CustomerID': range(1, 11),
        'Age': np.random.normal(35, 10, 10),
        'Income': np.random.normal(60000, 15000, 10),
        'PurchaseScore': np.random.randint(1, 100, 10).astype(float)
    }
    df = pd.DataFrame(data)
    
    # Introduce NaN (Missing Values)
    df.loc[2, 'Income'] = np.nan
    df.loc[7, 'Income'] = np.nan
    df.loc[4, 'PurchaseScore'] = np.nan
    
    # Introduce Outliers (Irregularities)
    df.loc[9, 'Age'] = 145.0  # Impossible age
    df.loc[0, 'Income'] = 9000000.0  # Extreme wealth anomaly
    
    return df

def detect_outliers_zscore(df, column, threshold=3.0):
    """Detects outliers mathematically using Z-Scores."""
    mean = df[column].mean()
    std = df[column].std()
    
    # Z = (X - Mean) / Standard_Deviation
    z_scores = (df[column] - mean) / std
    
    # Get absolute Z-scores
    abs_z_scores = np.abs(z_scores)
    outlier_mask = abs_z_scores > threshold
    return outlier_mask

def main():
    print_header("Step 1: Inspecting Messy Data")
    df = create_messy_data()
    print("Raw DataFrame:")
    print(df)
    
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    
    print_header("Step 2: Handling Irregularities (Outliers)")
    print("Using a lowered Z-Score threshold of 1.5 strictly for this small demonstration dataset.")
    # On large datasets, use 3.0. Here we use 1.5 because N=10 so standard deviation is heavily skewed.
    outlier_mask_age = detect_outliers_zscore(df, 'Age', threshold=1.5)
    outlier_mask_income = detect_outliers_zscore(df, 'Income', threshold=1.5)
    
    print("Detected Outliers in Age:\n", df[outlier_mask_age])
    print("\nDetected Outliers in Income:\n", df[outlier_mask_income])
    
    # Action: Remove rows where Age is an outlier
    df_cleaned = df[~outlier_mask_age].copy()
    # Action: Cap the Income outlier (Winsorization) instead of dropping it
    median_income = df_cleaned['Income'].median()
    df_cleaned.loc[df_cleaned['Income'] > 1000000, 'Income'] = median_income
    
    print("\nDataFrame after outlier treatment:")
    print(df_cleaned)
    
    print_header("Step 3: Addressing Missing Values")
    # Method A: Simple Imputation (Mean/Median) for PurchaseScore
    simple_imputer = SimpleImputer(strategy='median')
    df_cleaned['PurchaseScore'] = simple_imputer.fit_transform(df_cleaned[['PurchaseScore']])
    
    # Method B: Algorithmic Imputation (KNN) for Income
    # KNN looks at the closest neighbors based on Age and PurchaseScore to guess Income
    features_for_knn = ['Age', 'Income', 'PurchaseScore']
    knn_imputer = KNNImputer(n_neighbors=2)
    df_cleaned[features_for_knn] = knn_imputer.fit_transform(df_cleaned[features_for_knn])
    
    print("Final Refined DataFrame (No NaNs, No Extreme Outliers):")
    print(df_cleaned)
    print("\nCheck NaNs:\n", df_cleaned.isnull().sum())

if __name__ == "__main__":
    main()
