# Session 13 — Solutions

## 🟢 Solution 13.1: Pythonic Dispersion Metrics

```python
import pandas as pd

def print_dispersion_metrics(df, col_name):
    print(f"Metrics for {col_name}:")
    print(f"1. Std Dev:  {df[col_name].std():.2f}")
    print(f"2. Variance: {df[col_name].var():.2f}")
    print(f"3. Skewness: {df[col_name].skew():.2f}")
    print(f"4. Kurtosis: {df[col_name].kurtosis():.2f}")
```

## 🟢 Solution 13.2: Interpreting Skew
**Answers:**
1. A skewness of `2.8` is highly positive. Therefore, the data is **Right-Skewed** (the tail stretches to the right).
2. Because the data is right-skewed, the massive outliers on the right are pulling the average up. Therefore, the **Mean will be higher than the Median**.

---

## 🟡 Solution 13.3: Advanced Seaborn Grids

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')

# Generate the JointPlot
jplot = sns.jointplot(
    data=df, 
    x='total_bill', 
    y='tip', 
    hue='time', 
    palette='Set2'
)

# Tighten and Save
plt.tight_layout()
jplot.savefig('tips_jointplot.png', dpi=150)
print("Saved successfully!")
```

## 🟡 Solution 13.4: Finding the "Flattest" Feature

```python
import numpy as np

def find_flattest_feature(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate kurtosis
    kurtosis_series = numeric_df.kurtosis()
    
    # The lowest kurtosis (most negative) is the flattest/most platykurtic
    flattest_col = kurtosis_series.idxmin()
    lowest_score = kurtosis_series.min()
    
    print(f"Flattest feature is '{flattest_col}' with a score of {lowest_score:.2f}")
    return flattest_col
```

---

## 🔴 Solution 13.5: Generating an Automated HTML Profile

```python
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

# 1. Load data
df = sns.load_dataset('taxis')

# 2. Generate Profile Report
profile = ProfileReport(df, title="Taxis EDA Profiling Report", explorative=True)

# 3. Export to HTML
profile.to_file("report.html")
print("Report generated! Open report.html in Chrome/Firefox to view the automated Alerts.")
```
