# Session 01 — Solutions

## 🟢 Solution 1.1: Iris Classification

```python
"""
Solution 1.1: Iris Classification
──────────────────────────────────
A complete beginner classification example using the Iris dataset.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load data
iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {list(iris.target_names)}")

# Step 2: Split data — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training: {len(X_train)} | Test: {len(X_test)}")

# Step 3: Train a Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Bonus: Feature importance
import pandas as pd
importances = pd.Series(model.feature_importances_, index=iris.feature_names)
print("Feature Importances:")
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"  {feat}: {imp:.4f}")
```

**Expected Output**:
```
Accuracy: 100.00%
```

**Key Learnings**:
- `train_test_split` creates separate training and test sets
- `random_state` ensures reproducibility
- Decision Trees can perfectly classify Iris because it's a simple dataset
- High accuracy on Iris doesn't mean the model will work on complex data

---

## 🟢 Solution 1.2: Random State Experiment

```python
"""
Solution 1.2: Random State Experiment
──────────────────────────────────────
Demonstrates how different random splits affect model performance.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

# Experiment: 10 different random states
accuracies = []
random_states = range(10)

for rs in random_states:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rs
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)
    print(f"  random_state={rs}: Accuracy = {acc:.2%}")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"\nMean Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")

# Visualization
plt.figure(figsize=(10, 5))
bars = plt.bar(random_states, accuracies, color='#3498db', edgecolor='white')
plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.2%}')
plt.fill_between([-0.5, 9.5], mean_acc - std_acc, mean_acc + std_acc,
                  alpha=0.2, color='red', label=f'±1 Std: {std_acc:.2%}')
plt.xlabel('Random State')
plt.ylabel('Accuracy')
plt.title('Effect of Random State on Model Performance')
plt.legend()
plt.xticks(random_states)
plt.ylim(0.85, 1.05)
plt.tight_layout()
plt.show()

# Better approach: Cross-Validation
print("\n--- Better Approach: 5-Fold Cross-Validation ---")
model = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"CV Mean:   {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print("\n✅ Cross-validation gives a more reliable performance estimate!")
```

**Answers to Questions**:
1. **Why do results change?** Different random splits put different samples in train/test. Some splits are "easier" (test set happens to be simpler).
2. **What does high variance tell you?** The model is sensitive to which specific samples it sees. This suggests it might be overfitting or the dataset is too small.
3. **How does cross-validation solve this?** CV trains and tests on ALL data by rotating the test fold. This gives a more stable performance estimate.

---

## 🟡 Solution 1.4: Pipeline with GridSearchCV

```python
"""
Solution 1.4: Production Pipeline with GridSearchCV
─────────────────────────────────────────────────────
Demonstrates proper sklearn Pipeline usage with hyperparameter tuning.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge()),
])

# Define parameter grid
# Note: use "model__alpha" — double underscore connects step name to parameter
param_grid = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

# Create and fit GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

# Results
print(f"\nBest alpha: {grid_search.best_params_['model__alpha']}")
print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# Evaluate on test set
y_pred = grid_search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {test_rmse:.4f} (${test_rmse * 100000:,.0f} avg error)")
print(f"Test R²:   {test_r2:.4f}")

# Show all results
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print("\nAll Results:")
for _, row in results.iterrows():
    alpha = row['param_model__alpha']
    rmse = np.sqrt(-row['mean_test_score'])
    std = np.sqrt(row['std_test_score'])
    print(f"  alpha={alpha:>6.2f} → RMSE: {rmse:.4f} ± {std:.4f}")
```

---

## 🟡 Solution 1.5: Multi-Model Comparison

```python
"""
Solution 1.5: Multi-Model Comparison
──────────────────────────────────────
Compare 5 algorithms on California Housing with cross-validation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and split
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (α=1.0)": Ridge(alpha=1.0),
    "Lasso (α=0.01)": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv.mean())
    
    results.append({"Model": name, "Test RMSE": rmse, "R²": r2, "CV RMSE": cv_rmse})
    print(f"{name:25s} | RMSE: {rmse:.4f} | R²: {r2:.4f} | CV RMSE: {cv_rmse:.4f}")

df_results = pd.DataFrame(results).sort_values("Test RMSE")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_results)))
ax1.barh(df_results["Model"], df_results["Test RMSE"], color=colors)
ax1.set_title("Test RMSE (Lower = Better)")
ax1.set_xlabel("RMSE")

colors_r2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_results)))
ax2.barh(df_results["Model"], df_results["R²"], color=colors_r2)
ax2.set_title("R² Score (Higher = Better)")
ax2.set_xlabel("R²")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# Analysis
best = df_results.iloc[0]
print(f"\n📊 ANALYSIS:")
print(f"Best Model: {best['Model']}")
print(f"  RMSE: {best['Test RMSE']:.4f} (${best['Test RMSE']*100000:,.0f} avg error)")
print(f"  R²:   {best['R²']:.4f}")
print(f"\nGradient Boosting typically wins because it builds trees sequentially,")
print(f"with each tree correcting the errors of the previous ones. This additive")
print(f"approach captures complex non-linear relationships in the housing data.")
```

---

## 🔴 Solution 1.7: Bias Mitigation (Partial — for guidance)

```python
"""
Solution 1.7: Bias Mitigation Strategies
──────────────────────────────────────────
Implements and compares two bias mitigation approaches.
Strategy A: Remove protected + proxy features
Strategy B: Reweighting with class_weight='balanced'
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# [Load Adult Census dataset — same as 03_bias_audit.py]
# ... (data loading code omitted for brevity, see 03_bias_audit.py)

# ─── STRATEGY A: Remove Protected + Proxy Features ───
# Proxy features: relationship and marital_status correlate heavily with sex
protected_and_proxies = ["sex", "race", "native_country", "relationship", "marital_status"]
X_fair_A = X.drop(columns=protected_and_proxies, errors="ignore")

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X_fair_A, y, test_size=0.3, random_state=42, stratify=y
)
scaler_A = StandardScaler()
model_A = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model_A.fit(scaler_A.fit_transform(X_train_A), y_train_A)
y_pred_A = model_A.predict(scaler_A.transform(X_test_A))

# ─── STRATEGY B: Reweighting ───
# Give higher weight to underrepresented groups × outcomes
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler_B = StandardScaler()
X_train_B_s = scaler_B.fit_transform(X_train_B)

# Compute balanced sample weights
sample_weights = compute_sample_weight("balanced", y_train_B)

model_B = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model_B.fit(X_train_B_s, y_train_B, sample_weight=sample_weights)
y_pred_B = model_B.predict(scaler_B.transform(X_test_B))

# ─── COMPARISON TABLE ───
print(f"{'Metric':<25s} {'Original':>10s} {'Strategy A':>12s} {'Strategy B':>12s}")
print("-" * 60)
# ... (calculate and print metrics for all three models)
# Focus on: accuracy, DP ratio, EO difference
```

**Key Insight for Students**: 
- Strategy A reduces direct bias but may lose useful features
- Strategy B maintains all features but adjusts for class imbalance
- Neither is perfect — fairness requires ongoing monitoring and stakeholder input
- The "best" strategy depends on the specific fairness goals and stakeholder requirements

---

## 📌 Note to Students

Solutions are provided for learning purposes. The goal is not to copy them, but to:
1. **Try the exercise first** (spend at least 15-30 minutes)
2. **Compare your approach** with the solution
3. **Understand the reasoning** behind design choices
4. **Improve your solution** based on what you learned

> "The only way to learn programming is by programming." — Niklaus Wirth
