# Session 18 — Exercises

## 🟢 Beginner Exercises

### Exercise 18.1: Identifying Proxy Bias
**Objective**: Identify hidden vectors of discrimination.

**Instructions**:
A company builds an AI to predict if an applicant will default on a credit card. Ethically, they drop `Race` and `Gender` from the dataset. 
However, they left the column `Membership_In_Sorority_Or_Fraternity`. 
Write a script printing exactly which Protected Class (Race, Religion, Gender, Age) this column will act as a *Proxy* for, effectively retaining bias in the model.

### Exercise 18.2: The Four-Fifths Rules
**Objective**: Calculate the 80% legal threshold.

**Instructions**:
Your AI approves 60% of Men for a job interview.
Mathematically, what is the exact minimum percentage of Women it must approve to pass the Disparate Impact audit (`0.80`)?
Write a single line of python math that `print()`s this exact integer percentage.

---

## 🟡 Intermediate Exercises

### Exercise 18.3: Demographic Parity Function
**Objective**: Automate the Audit.

**Instructions**:
Write a python function `audit_model(df, protected_column, prediction_column)` that:
1. Calculates the Approval Rate of `df` where `protected_column == 1` (Majority).
2. Calculates the Approval Rate of `df` where `protected_column == 0` (Minority).
3. Divides minority by majority.
4. Returns `True` if the ratio is >= 0.80, and `False` if it is below.

Test it manually using simulated dummy data.

---

## 🔴 Advanced Exercises

### Exercise 18.4: Extracting False Positives Disparities
**Objective**: Uncover predictive imbalances beyond simple approval rates.

**Instructions**:
Sometimes, a Facial Recognition AI approves everyone equally, but its *Error Rates* are biased. 
Assume the AI accidentally tags innocent people as Criminals (False Positive). 
Write a script interpreting why it is catastrophically dangerous if the False Positive Rate (FPR) for minorities is 15% and the FPR for majorities is 1%.

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`).
- Answer conceptual questions efficiently in string prints.
- Name your files `exercise_18_X.py`.
