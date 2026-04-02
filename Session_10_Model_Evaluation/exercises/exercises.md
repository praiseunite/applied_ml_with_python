# Exercises: Model Evaluation (CV, Bias vs Variance)

---

## 🟢 BEGINNER: Bias vs Variance

**Question 1:**
You build an AI model to detect spam emails. On your Training Data, it achieves 99.8% accuracy. When you deploy it to real users (Testing Data), it achieves 45% accuracy. 
Is this model suffering from High Bias or High Variance? Explain why.

**Question 2:**
You build a linear regression model to predict housing prices, but the data is highly curved and nonlinear. The model achieves 30% accuracy on both the Training Data and the Testing Data. 
Is this model Underfitting or Overfitting? 

---

## 🟡 INTERMEDIATE: Cross-Validation Types

**Question 3:**
Explain the purpose of **Stratified** K-Fold Cross Validation compared to standard K-Fold. When is it absolutely mandatory to use Stratified?

**Question 4:**
Assume you are using Leave-One-Out Cross Validation (LOOCV) on a dataset with 5,000 rows. How many distinct times will you have to train the model from scratch? What is the main drawback of this approach?

---

## 🔴 ADVANCED: ROC Curve Mechanics

**Question 5:**
If a model generates an ROC-AUC score of 0.30, what does this mathematically imply about the model's predictions compared to random guessing (0.50)? How might you "fix" a model that consistently scores 0.30?

**Question 6:**
When looking at an ROC Curve specifically, where is the "perfect" point on the graph located? (Explain in terms of the X and Y axes).
