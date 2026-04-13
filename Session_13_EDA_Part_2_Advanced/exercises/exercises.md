# Session 13 — Exercises

## 🟢 Beginner Exercises

### Exercise 13.1: Pythonic Dispersion Metrics
**Objective**: Practice calling fast Pandas shape indicator functions.

**Instructions**:
Given a dataframe `df` with a numeric column `'HousePrice'`:
Write a script that clearly `prints` the following 4 metrics:
1. The standard deviation.
2. The Variance.
3. The Skewness.
4. The Kurtosis.

### Exercise 13.2: Interpreting Skew
**Objective**: Understand *why* we check for skewness.

**Instructions**:
If Exercise 13.1 output a Skewness of `2.8`, answer the following questions:
1. Is the data roughly symmetric, left-skewed, or right-skewed?
2. Would the Mean be higher or lower than the Median in this dataset?

---

## 🟡 Intermediate Exercises

### Exercise 13.3: Advanced Seaborn Grids
**Objective**: Use a `JointPlot` to uncover multivariate relationships.

**Instructions**:
We have the Seaborn `tips` dataset (`sns.load_dataset('tips')`). 
Write a script that creates a `sns.jointplot`:
1. X-axis should be `'total_bill'`.
2. Y-axis should be `'tip'`.
3. Use `hue='time'` to color coordinate by Lunch vs Dinner.
4. Save the plot to `tips_jointplot.png`.

### Exercise 13.4: Finding the "Flattest" Feature
**Objective**: Apply kurtosis to feature selection.

**Instructions**:
Write a function `find_flattest_feature(df)` that takes a pandas DataFrame, calculates the kurtosis for every numeric column, and returns the name of the column with the *lowest* (most negative) kurtosis score (meaning it has the flattest peak and shortest tails).

---

## 🔴 Advanced Exercises

### Exercise 13.5: Generating an Automated HTML Profile
**Objective**: Implement the industry standard `ydata-profiling` library.

**Instructions**:
1. Install `ydata-profiling`.
2. Load any pandas dataframe of your choice.
3. Write the 3 lines of python code required to generate a `ProfileReport` and export it to `report.html`.
4. Open the HTML file in your browser and find the "Alerts" section. What did the AI flag for you automatically?

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`).
- Answer conceptual questions in comments.
- Name your files `exercise_13_X.py`.
