# Session 15 — Solutions

## 🟢 Solution 15.1: Identifying Confounders

```python
explanation = """
No, Firetrucks do not cause fire damage.
The confounding variable is the "Severity/Size of the Fire".
A massive fire naturally pulls in a high number of firetrucks, AND naturally causes massive financial damage.
The severity of the fire causes both, creating a false correlation between firetrucks and damage.
"""
print(explanation)
```

## 🟢 Solution 15.2: The Alpha Threshold

```python
print("1. Is this result statistically significant?")
print("Answer: NO. The p-value (0.06) is greater than the standard alpha threshold of 0.05.")
print("2. Should the business roll out the new feature?")
print("Answer: NO. Because p > 0.05, we must assume the Null Hypothesis is true, meaning the new feature had no real impact beyond random luck.")
```

---

## 🟡 Solution 15.3: Implementing `scipy` T-Tests

```python
import numpy as np
from scipy import stats

group_a = np.random.normal(50, 10, 100)
group_b = np.random.normal(55, 12, 100)

def evaluate_experiment(control, treatment):
    t_stat, p_val = stats.ttest_ind(treatment, control)
    print(f"P-Value calculated: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Result: Deploy Treatment")
    else:
        print("Result: Stop Experiment")

evaluate_experiment(group_a, group_b)
```

---

## 🔴 Solution 15.4: Calculating Propensity Scores

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Assuming df exists... (Dummy df created for testing structure)
df = pd.DataFrame({
    'Age': [25, 45, 60],
    'BMI': [22.0, 28.5, 30.1],
    'Blood_Pressure': [110, 130, 145],
    'Received_Medicine': [0, 1, 1]
})

# 1. Initialize model
model = LogisticRegression()

# 2. Extract Confounders
confounders = ['Age', 'BMI', 'Blood_Pressure']

# 3. Fit the model to predict the Treatment
model.fit(df[confounders], df['Received_Medicine'])

# 4. Extract Propensity Scores (The probability column [:, 1])
df['Propensity_Score'] = model.predict_proba(df[confounders])[:, 1]

print("Propensity Scores successfully applied to the DataFrame!")
print(df[['Received_Medicine', 'Propensity_Score']])
```
