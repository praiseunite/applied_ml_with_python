# Session 17 — Solutions

## 🟢 Solution 17.1: Model Categorization

```python
print("1. Linear Regression: White Box (You can directly read the mathematical weights: y = mx + b)")
print("2. Deep Neural Network: Black Box (Massive matrices calculating trillions of floating point derivatives)")
print("3. Decision Tree (Depth 3): White Box (Easily understandable If/Else graph)")
print("4. XGBoost: Black Box (Hundreds of trees correcting each other mathematically is unreadable)")
print("5. K-Nearest Neighbors: White Box (It literally just points to the closest user data points)")
```

## 🟢 Solution 17.2: Understanding SHAP Colors

```python
answer = """
The correct answer is B.
- Blue dots mean the numerical input was LOW (e.g. Age was low).
- Being on the FAR RIGHT means the SHAP prediction impact was highly POSITIVE.
- Therefore, a Low input drove the prediction Higher.
"""
print(answer)
```

---

## 🟡 Solution 17.3: Controlling Genetic Generations (TPOT)

```python
from tpot import TPOTClassifier

# We explicitly limit max evolutionary cycles to prevent CPU burn
clf = TPOTClassifier(generations=2, population_size=15, verbosity=2)

# Assume X_train, y_train exist...
# clf.fit(X_train, y_train)
```

---

## 🔴 Solution 17.4: Extracting the Winning Code

```python
# Assuming clf is the completed TPOTClassifier from Exercise 17.3
# clf.export('company_optimal_pipeline.py')
print("The command is: clf.export('company_optimal_pipeline.py')")
```
