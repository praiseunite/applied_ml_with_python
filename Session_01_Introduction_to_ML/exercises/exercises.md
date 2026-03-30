# Session 01 — Exercises

## 🟢 Beginner Exercises

### Exercise 1.1: Iris Classification

**Objective**: Build your first classification model.

**Instructions**:
1. Load the Iris dataset using `sklearn.datasets.load_iris()`
2. Split the data into 80% training and 20% test sets
3. Train a `DecisionTreeClassifier`
4. Calculate and print the accuracy
5. Print the classification report

**Starter Code**:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load data
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split data (TODO: fill in the parameters)
X_train, X_test, y_train, y_test = train_test_split(
    # YOUR CODE HERE
)

# Step 3: Train model (TODO)
model = # YOUR CODE HERE

# Step 4: Predict and evaluate (TODO)
y_pred = # YOUR CODE HERE

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Expected Output**: Accuracy around 95-100%

---

### Exercise 1.2: Random State Experiment

**Objective**: Understand how random splits affect model performance.

**Instructions**:
1. Using the Iris dataset from Exercise 1.1
2. Train DecisionTreeClassifier with 10 different `random_state` values (0-9)
3. Record the accuracy for each
4. Calculate the mean and standard deviation of accuracies
5. Plot the results as a bar chart

**Questions**:
- Why do results change with different random states?
- What does high variance tell you about your model?
- How can cross-validation solve this problem?

**Expected**: You should see accuracy varying between ~90-100%

---

### Exercise 1.3: ML in Your Daily Life

**Objective**: Identify real-world ML applications.

**Instructions**: 
List 5 ML applications you use daily. For each, identify:

| # | Application | ML Type | What It Predicts | Data It Uses |
|---|------------|---------|-----------------|--------------|
| 1 | (example) YouTube Recommendations | Recommendation System | Videos you'll like | Watch history, likes, demographics |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

---

## 🟡 Intermediate Exercises

### Exercise 1.4: Pipeline with GridSearchCV

**Objective**: Build a production-grade ML pipeline.

**Instructions**:
1. Load the California Housing dataset
2. Create a `Pipeline` with `StandardScaler` + `Ridge` regression
3. Use `GridSearchCV` to tune the `alpha` parameter: `[0.01, 0.1, 1.0, 10.0, 100.0]`
4. Print the best parameters and best cross-validation score
5. Evaluate the best model on the test set

**Starter Code**:
```python
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# TODO: Create pipeline
pipeline = Pipeline([
    # YOUR CODE HERE
])

# TODO: Define parameter grid
param_grid = {
    # YOUR CODE HERE
}

# TODO: Create and fit GridSearchCV
grid_search = GridSearchCV(
    # YOUR CODE HERE
)
grid_search.fit(X_train, y_train)

# TODO: Print results
print(f"Best alpha: {grid_search.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# Evaluate on test set
y_pred = grid_search.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred):.4f}")
```

---

### Exercise 1.5: Multi-Model Comparison

**Objective**: Compare 5 algorithms and determine which is best for a given problem.

**Instructions**:
1. Load the California Housing dataset
2. Train these 5 models: LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting
3. Use 5-fold cross-validation for each
4. Create a comparison table and bar chart
5. Write a paragraph explaining which model is best AND why

**Deliverable**: A complete script with visualization and written analysis.

---

### Exercise 1.6: Bias Audit by Race

**Objective**: Extend the bias audit to analyze racial disparities.

**Instructions**:
1. Using the code from `03_bias_audit.py` as a starting point
2. Calculate demographic parity across ALL racial groups in the Adult dataset
3. Calculate equalized odds (TPR and FPR) for each racial group
4. Create a visualization showing disparities
5. Write a 200-word analysis of your findings

**Questions**:
- Which racial group has the highest/lowest positive prediction rate?
- Does the model exhibit equal true positive rates across groups?
- What could explain the disparities you observe?

---

## 🔴 Advanced Exercises

### Exercise 1.7: Bias Mitigation Implementation

**Objective**: Implement and compare bias mitigation strategies.

**Instructions**:
1. Train the original model from `03_bias_audit.py`
2. Implement TWO mitigation strategies:
   - **Strategy A**: Remove protected attributes + highly correlated proxies
   - **Strategy B**: Reweighting — assign higher weights to underrepresented groups
3. For each strategy, measure:
   - Overall accuracy (should stay reasonable)
   - Demographic parity ratio
   - Equalized odds difference
4. Create a comparison table:

| Metric | Original | Strategy A | Strategy B |
|--------|----------|------------|------------|
| Accuracy | | | |
| DP Ratio (Gender) | | | |
| EO Diff (Gender) | | | |

**Hint for Strategy B (Reweighting)**:
```python
# Calculate sample weights inversely proportional to group size × outcome
from sklearn.utils.class_weight import compute_sample_weight

# Method 1: Use sample_weight parameter in model.fit()
weights = compute_sample_weight("balanced", y_train)
model.fit(X_train, y_train, sample_weight=weights)
```

---

### Exercise 1.8: Model Card

**Objective**: Write a professional model card.

**Instructions**:
Create a Model Card for the diabetes prediction model following [Google's Model Card template](https://modelcards.withgoogle.com/about). Your card should include:

1. **Model Details**: Name, version, type, developers, date
2. **Intended Use**: Primary use, out-of-scope uses
3. **Training Data**: Description, preprocessing, size
4. **Evaluation Data**: Description, size
5. **Metrics**: Performance metrics with confidence intervals
6. **Ethical Considerations**: Potential harms, limitations
7. **Caveats and Recommendations**: What users should know

**Format**: Create this as a Markdown file (`model_card.md`)

---

### Exercise 1.9: EU AI Act Classification

**Objective**: Apply regulatory knowledge to an ML system.

**Instructions**:
1. Read about the [EU AI Act risk categories](https://artificialintelligenceact.eu/)
2. For EACH of these ML systems, classify the risk level and explain why:

| System | Risk Level | Justification |
|--------|-----------|---------------|
| Spam email filter | | |
| Resume screening tool | | |
| Medical diagnosis AI | | |
| Social scoring system | | |
| Movie recommendation | | |
| Autonomous vehicle AI | | |
| Facial recognition for law enforcement | | |
| Customer chatbot | | |

3. For ONE high-risk system above, list all requirements it would need to meet under the EU AI Act.

---

## 📝 Submission Guidelines

- Submit all code as `.py` files or Jupyter notebooks (`.ipynb`)
- Include comments explaining your reasoning
- All code must run without errors
- Include visualizations where requested
- Written analyses should be thoughtful and specific (not generic)
