# Session 17: Automated Machine Learning & Transparent Models

Welcome to **Session 17** of the Applied Machine Learning Using Python curriculum!

As algorithms become more powerful, they also become more complex. In this session, you will learn how to let AI build its own pipelines using **AutoML**, and more importantly, how to force those complex "Black-Box" algorithms to explain themselves to you mathematically.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Define AutoML and explain its workflow/significance.
2. Build automated ML models and pipelines using Genetic Algorithms (TPOT).
3. Describe transparency in ML (White-Box vs Black-Box models).
4. Extract mathematical explainability using the **SHAP** framework.

---

## 📖 Part 1: Automated Machine Learning (AutoML)

Data Scientists often waste weeks manually testing different algorithms (Random Forest, SVM, XGBoost) and twisting hundreds of hyperparameters to see which combination works best. **AutoML** automates this entirely.

Instead of writing manual training loops, we give an AutoML framework (like `TPOT` or `PyCaret`) our raw data. The framework uses **Genetic Algorithms** to spawn thousands of random ML pipelines (e.g., *Pipeline A: StandardScale -> PCA -> RandomForest*). It evaluates them, kills the bad pipelines, "breeds" the best ones together, and eventually hands you the absolute mathematically perfect Python Pipeline for your dataset.

### Applications of AutoML
- **Rapid Prototyping:** Instantly establishing a "Baseline" score before human engineers try to beat it.
- **Pipeline Optimization:** Finding obscure mathematical combinations of Scalers and Dimensionality Reductions that humans would never think to pair together.

---

## 📖 Part 2: Transparency in Machine Learning

If an algorithm rejects a user's mortgage application, the user has a legal right to ask *"Why?"*.

- **White-Box Models:** Inherently transparent. (e.g., A simple Decision Tree or Linear Regression). You can literally read the `if/else` statements or the Weights to understand exactly why a decision was made.
- **Black-Box Models:** Highly complex. (e.g., 500-layer Neural Networks, extreme XGBoost Ensembles). These models are incredibly accurate, but no human alive can read the raw math and explain *why* the model made a specific prediction.

### The SHAP Solution
To solve the Black-Box problem, the ML industry adopted **SHAP (SHapley Additive exPlanations)**. 
Derived from Cooperative Game Theory, SHAP mathematically forces *any* algorithm to assign an exact Numerical Contribution Value to every single feature for every single prediction. 

It allows you to say: *"The AI rejected your mortgage because your Credit Score subtracted 0.4 points from your approval chance, and your Income level only added 0.1 points."*

---

## 🚀 Hands-On: Session Code Files

In the `code/` directory, you will find scripts demonstrating these Applied Python concepts:
1. **`01_automl_pipeline.py`**: Executes `TPOTClassifier` on breast cancer data, automatically searching dozens of algorithms and exporting the final optimal pipeline as raw Python code.
2. **`02_model_transparency_shap.py`**: Trains a highly complex Random Forest, and uses the `shap` library to generate a Waterfall chart breaking down the exact mathematical logic behind a specific user prediction.

---
*© 2024 Aptech Limited — For Educational Use*
