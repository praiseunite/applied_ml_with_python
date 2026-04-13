# Session 17 — Exercises

## 🟢 Beginner Exercises

### Exercise 17.1: Model Categorization
**Objective**: Differentiate between Transparent and Black-Box models.

**Instructions**:
Write a python script that prints either "White Box" or "Black Box" for each of the following algorithms:
1. Linear Regression
2. Deep Neural Network
3. Standard Decision Tree (Depth = 3)
4. XGBoost Ensemble (1000 estimators)
5. K-Nearest Neighbors

### Exercise 17.2: Understanding SHAP Colors
**Objective**: Interpret the mechanics of a SHAP Summary Plot.

**Instructions**:
In a SHAP summary plot, if you see a large cluster of **BLUE dots** on the **FAR RIGHT** side of the horizontal zero-line, what does that mean?
A) High numerical inputs caused the probability metric to rise.
B) Low numerical inputs caused the probability metric to rise.
Write a python script that prints your answer.

---

## 🟡 Intermediate Exercises

### Exercise 17.3: Controlling Genetic Generations (TPOT)
**Objective**: Build a safe, fast pipeline search.

**Instructions**:
You have been given a dataset (`X_train, y_train`). 
Your manager wants you to use `TPOTClassifier` to search for an optimal pipeline. However, she explicitly warns you: "Do NOT let this run for 10 hours and burn our CPU credits."
Write the exact 3 lines of python code required to initialize and `.fit()` a `TPOTClassifier` limited to **only 2 generations** with a population size of **15**.

---

## 🔴 Advanced Exercises

### Exercise 17.4: Extracting the Winning Code
**Objective**: Capture the AutoML results.

**Instructions**:
After running your pipeline from Exercise 17.3, what python function must you call from the `tpot` object to automatically write out the champion Scikit-Learn code as a `.py` file?
Write a script demonstrating this single function call, saving the file as `company_optimal_pipeline.py`.

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`).
- Name your files `exercise_17_X.py`.
