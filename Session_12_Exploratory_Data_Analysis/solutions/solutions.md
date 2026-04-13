# Session 12 — Solutions

## 🟢 Solution 12.1: Basic Summary Statistics

```python
import pandas as pd

def quick_summary(df, column_name):
    print(f"Summary for column: {column_name}")
    print(f"Mean: {df[column_name].mean()}")
    print(f"Median: {df[column_name].median()}")
    print(f"Std Dev: {df[column_name].std()}")
    print(f"Missing Values: {df[column_name].isnull().sum()}")
```

---

## 🟢 Solution 12.2: Identifying Patterns visually

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Using seaborn to find trends between Engine Size and CO2 Emissions
sns.scatterplot(data=df, x='EngineSize', y='CO2Emissions', alpha=0.6, color='blue')
plt.title("Correlation between Engine Size and Emissions")
plt.show()
```

---

## 🟡 Solution 12.3: Z-Score Outlier Removal

```python
import numpy as np

def remove_zscore_outliers(df, column):
    mean = df[column].mean()
    std = df[column].std()
    
    # Calculate Z-Scores
    df['Z_score'] = (df[column] - mean) / std
    
    # Filter the DataFrame
    df_cleaned = df[(df['Z_score'] > -3.0) & (df['Z_score'] < 3.0)]
    
    # Drop the temporary column
    df_cleaned = df_cleaned.drop(columns=['Z_score'])
    
    return df_cleaned
```

---

## 🟡 Solution 12.4: The IQR Cap (Winsorization)

```python
def cap_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap upper outliers
    df.loc[df[column] > upper_bound, column] = upper_bound
    # Cap lower outliers
    df.loc[df[column] < lower_bound, column] = lower_bound
    
    return df
```

---

## 🔴 Solution 12.5: The KNN Imputer

```python
from sklearn.impute import KNNImputer

# Define features used to determine "nearest neighbors"
features = ['Age', 'ExperienceYears', 'Salary']

# Initialize imputer
knn_imputer = KNNImputer(n_neighbors=5)

# Execute imputation
# Note: KNNImputer returns a numpy array, so we overwrite the Pandas DataFrame dynamically
df[features] = knn_imputer.fit_transform(df[features])

print("Missing salaries have been resolved using the 5 most demographically similar rows!")
```

---

## 🔴 Solution 12.6: Refining Optimal Presentations

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Initial plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='Department', y='Budget', palette='viridis')

# 1. Title formatting
plt.title("Annual Budget Allocation by Department", weight='bold', fontsize=16)

# 2. Despine borders
sns.despine(top=True, right=True)

# 3. Horizontal Gridlines only
plt.grid(axis='y', alpha=0.3)

# 4. Rotate X-ticks for readability
plt.xticks(rotation=45, ha='right')

# Tight layout ensures text doesn't get cut off on export
plt.tight_layout()
plt.show()
```
