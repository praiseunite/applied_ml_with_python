# Session 09: Ensemble Learning & Model Evaluation

Welcome to **Session 09** of the Applied Machine Learning Using Python curriculum!

Often in machine learning, a single algorithm isn't enough to capture the full complexity of a dataset without overfitting. To solve this, we use **Ensemble Learning**—combining multiple weak models into a single, highly accurate "super-model." 

In this session, you will learn the core techniques behind the models that win almost every Kaggle tabular data competition: Random Forests, XGBoost, and LightGBM.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Explain the "Wisdom of the Crowd" concept through **Voting** classifiers and regressors.
2. Differentiate between **Bagging** (Parallel) and **Boosting** (Sequential) ensembles.
3. Understand the mathematical mechanics of Random Forests and XGBoost.
4. Properly evaluate models using advanced metrics like Cross-Validation, ROC-AUC, and F1-Scores.
5. Deploy state-of-the-art boosting models using Scikit-Learn and external libraries.

---

## 🟢 BEGINNER: The Wisdom of the Crowd

Imagine asking a single person to guess the exact weight of a cow at a fair. Their guess might be completely wrong. But if you ask 1,000 random people and take the average of their guesses, the final answer will be surprisingly close to the true weight!

This is the principle behind **Ensemble Learning**. 

### 1. Voting Classifiers
Instead of training one Logistic Regression model, what if we train:
1. A Logistic Regression model
2. A Decision Tree
3. A Support Vector Machine (SVM)

We can then have them "vote" on the outcome.
- **Hard Voting (Majority Rules):** If 2 models predict "Cat" and 1 predicts "Dog", the final prediction is "Cat".
- **Soft Voting (Confidence Based):** Averages the actual predicted probabilities of each model to make a highly confident final decision.

---

## 🟡 INTERMEDIATE: Bagging vs. Boosting

If we want to build an ensemble from the exact same algorithm (e.g., 100 Decision Trees), we need a way to ensure the trees are actually different. If they are identical, their voting is useless!

### Bagging (Bootstrap Aggregating)
Bagging trains hundreds of trees **in parallel**.
- **How it works:** It takes random sub-samples of the dataset with replacement (Bootstrapping) and trains a feature-restricted Decision Tree on each sub-sample. 
- **The Result:** Because each tree looks at a slightly different piece of the data, they make uncorrelated errors. When averaged together (Aggregated), the variance plummets.
- **Algorithm:** **Random Forest**.

### Boosting
Boosting trains trees **sequentially**.
- **How it works:** Tree #1 is trained and makes mistakes. Tree #2 is then trained *specifically* to fix the mistakes of Tree #1. Tree #3 is trained to fix the mistakes of Tree #2, and so on.
- **The Result:** It intensely reduces bias and gets highly accurate, but is prone to overfitting if you add too many trees.
- **Algorithms:** AdaBoost, Gradient Boosting, XGBoost, LightGBM.

---

## 🔴 ADVANCED: XGBoost & Advanced Evaluation

### XGBoost (eXtreme Gradient Boosting)
XGBoost revolutionized tabular machine learning. It builds upon standard Gradient Boosting but introduces advanced regularization (L1/L2 penalties on leaf weights) and hardware-level optimizations (cache-awareness, exact greedy algorithms for split finding).
- **Core hyper-parameters:** `learning_rate` (eta), `max_depth`, `n_estimators`, `subsample`.

### Proper Model Evaluation
Accuracy is useless for imbalanced data (e.g., predicting a rare disease). We must use advanced metrics:
- **Cross-Validation:** Splitting the data into $K$ folds and validating training stability across all of them.
- **ROC Curve (Receiver Operating Characteristic):** Plots the True Positive Rate against the False Positive Rate at different decision thresholds.
- **AUC (Area Under the Curve):** A single metric from 0.0 to 1.0 representing the ROC curve's performance. An AUC of 0.5 is random guessing; 1.0 is perfect prediction.

---

## 💻 HANDS-ON LAB: Ensembles & Evaluation

Open the Jupyter Notebook provided in this session: `notebooks/01_Ensemble_Methods.ipynb`.

**What you will do:**
1. Load a structured classification dataset (e.g., Breast Cancer dataset).
2. Train a baseline single Decision Tree.
3. Train a Random Forest and compare the huge leap in accuracy.
4. Train an XGBoost model and tune its hyperparameters.
5. Generate an ROC Curve showing all three models simultaneously!

---

## 📊 PORTFOLIO TASK

For your **Credit Risk Assessment** (Project 3) or **Stock Forecasting** (Project 2) portfolio projects, a single algorithm will not suffice. Implement an ensemble method (specifically Random Forest or XGBoost) and present the **ROC-AUC** score as your primary success metric in your Hugging Face space dashboard.

---

## 📚 FURTHER READING
- **XGBoost Paper:** "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- **Scikit-Learn Guide:** Model Evaluation - https://scikit-learn.org/stable/modules/model_evaluation.html

---
*© 2024 Aptech Limited — For Educational Use*
