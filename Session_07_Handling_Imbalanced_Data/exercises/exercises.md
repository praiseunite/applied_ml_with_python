# Exercises: Handling Imbalanced Data & PCA

---

## 🟢 BEGINNER: Feature Selection

**Question 1:**
Explain the core difference between a **Filter Method** and a **Wrapper Method** for Feature Selection. Which one is computationally faster?

**Question 2:**
You are building an ML model to detect a rare disease (1 in 100,000 cases). If your model just outputs "Healthy" for every single patient, what is its Accuracy? Why is this a problem?

---

## 🟡 INTERMEDIATE: PCA & Imbalanced Sets

**Question 3:**
You have a 500-column tabular dataset. You apply PCA and condense it down to 5 columns. Will you still be able to mathematically read exactly what "Column #3" means in the real world (e.g., "Age" or "Salary")? Why or why not?

**Question 4:**
Explain the main danger of using **Random Under-sampling** on a dataset with 1,000,000 normal transactions and 500 fraud transactions. 

---

## 🔴 ADVANCED: SMOTE Application

**Question 5:**
When applying SMOTE (Synthetic Minority Over-sampling Technique), is it mathematically valid to apply it to your entire dataset *before* you split it into Training and Testing sets? Explain exactly why this is a catastrophic mistake.

**Question 6:**
Describe briefly how the SMOTE algorithm generates synthetic data. Does it just duplicate existing rows?
