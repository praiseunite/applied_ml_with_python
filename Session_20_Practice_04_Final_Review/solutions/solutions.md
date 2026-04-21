# Session 20 — Solutions

## Full Pipeline Reference

The complete solution is implemented in `code/02_full_pipeline.py`. Run it with:

```bash
python code/01_generate_data.py   # Generate the dataset first
python code/02_full_pipeline.py   # Run the full pipeline
```

Below are the key solution snippets for the guided notebook phases.

---

## Phase 1: EDA — Key Findings

```python
# Class balance check
print(df['Attrition'].value_counts())
# Expected: ~80% Stayed (0), ~20% Left (1) — IMBALANCED

# Missing values
print(df.isnull().sum())
# Expected: ~5% missing in Age, MonthlySalary, OvertimeHours, 
# JobSatisfaction, PerformanceRating

# Gender pay gap (intentional bias in the data)
print(df.groupby('Gender')['MonthlySalary'].mean())
# Male mean will be ~8% higher than Female mean
```

---

## Phase 2: Data Cleaning

```python
# Median imputation (robust to outliers)
for col in ['Age', 'MonthlySalary', 'OvertimeHours', 'JobSatisfaction', 'PerformanceRating']:
    df[col].fillna(df[col].median(), inplace=True)

# IQR outlier capping for OvertimeHours
Q1 = df['OvertimeHours'].quantile(0.25)
Q3 = df['OvertimeHours'].quantile(0.75)
IQR = Q3 - Q1
df['OvertimeHours'] = df['OvertimeHours'].clip(upper=Q3 + 1.5 * IQR)
```

---

## Phase 3: Feature Engineering

```python
# Encode Gender as binary
df['Gender_Code'] = df['Gender'].map({'Male': 1, 'Female': 0})

# One-hot encode Department (drop_first avoids multicollinearity)
dept_dummies = pd.get_dummies(df['Department'], prefix='Dept', drop_first=True)
df = pd.concat([df, dept_dummies], axis=1)

# SMOTE for class balance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
```

---

## Phase 4: Model Training & Evaluation

```python
# Random Forest (Bagging — reduces variance)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, 
                            class_weight='balanced', random_state=42)
rf.fit(X_train_bal, y_train_bal)

# Gradient Boosting (Boosting — reduces bias)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, 
                                 learning_rate=0.1, random_state=42)
gb.fit(X_train_bal, y_train_bal)

# Select by ROC-AUC (NOT accuracy!)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
best = rf if rf_auc >= gb_auc else gb
```

**Why ROC-AUC, not Accuracy?**
If 80% of employees stay, a model that always predicts "Stay" gets 80% accuracy but catches zero at-risk employees. ROC-AUC measures discrimination ability independent of class balance.

---

## Phase 5: Explainability

```python
# Feature importance from the ensemble model
importances = best.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

for i in range(10):
    idx = sorted_idx[i]
    print(f"  {i+1}. {feature_cols[idx]:25s} {importances[idx]:.4f}")

# Expected top predictors: JobSatisfaction, OvertimeHours, MonthlySalary,
# WorkLifeBalance, PromotionsLast5Yr
```

---

## Phase 6: Ethical Audit

```python
# Disparate Impact by Gender
gender_idx = feature_cols.index('Gender_Code')
test_genders = X_test[:, gender_idx]

male_rate = y_pred[test_genders == 1].mean()
female_rate = y_pred[test_genders == 0].mean()

# The Four-Fifths Rule
ratio = min(male_rate, female_rate) / max(male_rate, female_rate)
print(f"Disparate Impact: {ratio:.3f}")

# If ratio < 0.80: INVESTIGATE
# MonthlySalary likely acts as a PROXY for Gender because
# of the intentional pay gap we injected in the data generator.
# This is Proxy Bias (Session 18, Exercise 18.1).
```

---

## Key Takeaway

This project proves that building an ML model is only **Step 6 of 10** in a real pipeline. Without the other 9 steps (EDA, cleaning, balancing, evaluation, explainability, ethical audit), you're shipping a liability — not a product.
