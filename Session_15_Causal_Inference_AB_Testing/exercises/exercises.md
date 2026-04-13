# Session 15 — Exercises

## 🟢 Beginner Exercises

### Exercise 15.1: Identifying Confounders
**Objective**: Understand the difference between correlation and causation conceptually.

**Instructions**:
A study finds a massive positive correlation between "Number of Firetrucks at a Fire" and "Amount of Fire Damage in Dollars."
Does sending more firetrucks to a scene cause more damage?
1. Write a Python script that `print()`s a multi-line string explaining what the hidden **Confounding Variable** is that explains this correlation.

### Exercise 15.2: The Alpha Threshold
**Objective**: Learn standard hypothesis thresholds.

**Instructions**:
You run an A/B test. The P-Value comes back as `0.06`. 
Assuming a standard Alpha threshold of 0.05, print a Python statement answering:
1. Is this result statistically significant?
2. Should the business roll out the new feature?

---

## 🟡 Intermediate Exercises

### Exercise 15.3: Implementing `scipy` T-Tests
**Objective**: Build basic A/B test analysis logic.

**Instructions**:
You are given two Numpy Arrays representing money spent per user:
`group_a = np.random.normal(50, 10, 100)`
`group_b = np.random.normal(55, 12, 100)`

Write a function `evaluate_experiment(group_a, group_b)` that:
1. Calculates the t-statistic and the p-value using `scipy.stats.ttest_ind`.
2. Prints "Deploy Treatment" if the p-value is less than 0.05.
3. Prints "Stop Experiment" if the p-value is greater than or equal to 0.05.

---

## 🔴 Advanced Exercises

### Exercise 15.4: Calculating Propensity Scores
**Objective**: Integrate Scikit-Learn Logistic Regression to calculate probabilities.

**Instructions**:
You have a DataFrame `df` with columns `['Age', 'BMI', 'Blood_Pressure', 'Received_Medicine']`. The data is observational (not an RCT).
Write a Python script that:
1. Initializes a Logistic Regression model.
2. Trains the model using Age, BMI, and Blood Pressure to predict `Received_Medicine`.
3. Creates a new column in `df` called `Propensity_Score` that contains the predicted probability of receiving the medicine, generated via `model.predict_proba()`.

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`).
- Answer conceptual questions efficiently in string prints.
- Name your files `exercise_15_X.py`.
