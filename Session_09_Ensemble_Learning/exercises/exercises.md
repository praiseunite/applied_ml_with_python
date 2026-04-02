# Exercises: Ensemble Learning & Evaluation

Complete these exercises to test your knowledge of Voting, Bagging, Boosting, and Classification Metrics.

---

## 🟢 BEGINNER: Core Concepts

**Question 1:**
What is the difference between "Hard Voting" and "Soft Voting" in an Ensemble Classifier?

**Question 2:**
Does a Random Forest train its Decision Trees in sequence (one after the other) or in parallel (all at the same time)?

**Question 3:**
If your model simply guesses "Not Fraud" for every transaction in a credit card dataset where 99% of transactions are legitimate, what is its Accuracy? Why is Accuracy a bad metric here?

---

## 🟡 INTERMEDIATE: Mechanics & implementation

**Question 4:**
Explain the concept of "Bootstrapping" in Bagging. Why is it necessary when building a Random Forest?

**Question 5:**
In Gradient Boosting, what role does the "learning rate" (or shrinkage) parameter play? What happens if you make it too high?

**Question 6:**
You are evaluating an XGBoost model. The ROC-AUC score is 0.51. What does this score tell you about the model's predictive capabilities?

---

## 🔴 ADVANCED: Hyperparameters & Optimization

**Question 7:**
Unlike a Random Forest, adding more trees (`n_estimators`) to a Gradient Boosting model can actually cause it to perform worse on testing data. Explain why this happens mathematically.

**Question 8:**
Write a small Python code snippet using `scikit-learn` that calculates the 5-fold cross-validation ROC-AUC score for a Random Forest classifier.
```python
# Your code here
```
