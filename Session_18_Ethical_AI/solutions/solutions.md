# Session 18 — Solutions

## 🟢 Solution 18.1: Identifying Proxy Bias

```python
explanation = """
The column 'Membership_In_Sorority_Or_Fraternity' explicitly separates users 
into Sororities (100% Female) and Fraternities (100% Male). 
By leaving this column in the dataset, the AI has a perfect PROXY 
for GENDER. The AI can still maliciously learn to discriminate against Women.
"""
print(explanation)
```

## 🟢 Solution 18.2: The Four-Fifths Rules

```python
# To pass the 0.80 ratio: 
# (Minority Rate) / 60% = 0.80
# Minority Rate = 60 * 0.80

min_acceptable_rate = 60 * 0.8
print(f"The AI must approve Women at a rate of exactly {min_acceptable_rate}% to pass.")
```

---

## 🟡 Solution 18.3: Demographic Parity Function

```python
import pandas as pd
import numpy as np

def audit_model(df, protected_column, prediction_column):
    maj_rate = df[df[protected_column] == 1][prediction_column].mean()
    min_rate = df[df[protected_column] == 0][prediction_column].mean()
    
    impact_ratio = min_rate / maj_rate
    print(f"Ratio evaluated: {impact_ratio:.2f}")
    
    return impact_ratio >= 0.80

# Dummy Test
df_test = pd.DataFrame({
    'Race': [1, 1, 0, 0, 0],
    'AI_Approve': [1, 1, 0, 1, 0] # Maj=100%, Min=33%
})

is_fair = audit_model(df_test, 'Race', 'AI_Approve')
print(f"Model is Fair: {is_fair}")
```

---

## 🔴 Solution 18.4: Extracting False Positives Disparities

```python
print("If the False Positive Rate is 15x higher for a minority demographic, it means")
print("the AI is 15x more likely to ruin an innocent minority person's life by falsely tagging")
print("them as a criminal compared to the majority, leading to massive civil rights violations.")
print("Approval parity is not enough. You must also check False Positive Parity.")
```
