# Session 12 — Exercises

## 🟢 Beginner Exercises

### Exercise 12.1: Basic Summary Statistics
**Objective**: Practice using Pandas to summarize a dataset's distributions.

**Instructions**:
Write a function `quick_summary(df, column_name)` that takes a pandas DataFrame and a numerical column name. It should output:
1. The Mean.
2. The Median.
3. The Standard Deviation.
4. The Count of `NaN` (missing) values in that column.

---

### Exercise 12.2: Identifying Patterns visually
**Objective**: Build basic Seaborn plots to search for underlying trends.

**Instructions**:
Write code to generate a **Scatter Plot** comparing `'EngineSize'` on the X-axis to `'CO2Emissions'` on the Y-axis. 
*Hint: Use `sns.scatterplot()`*.

---

## 🟡 Intermediate Exercises

### Exercise 12.3: Z-Score Outlier Removal
**Objective**: Filter data using standard deviations mathematically.

**Instructions**:
A dataset has a column called `'Salary'`. You want to drop any row where the salary is mathematically an outlier.
1. Write a function `remove_zscore_outliers(df, column)` that calculates the Z-Score for every value.
2. Filter the `df` to keep only rows where the Z-Score is between `-3.0` and `+3.0`.
3. Return the cleaned DataFrame.

---

### Exercise 12.4: The IQR Cap (Winsorization)
**Objective**: Avoid data deletion by capping outliers automatically.

**Instructions**:
Write a function `cap_iqr(df, column)`:
1. Find the 25th percentile (Q1) and 75th percentile (Q3).
2. Calculate the IQR `(Q3 - Q1)`.
3. Calculate the boundaries: `Lower = Q1 - (1.5 * IQR)`, `Upper = Q3 + (1.5 * IQR)`.
4. Replace all values exceeding `Upper` with the `Upper` value.
5. Replace all values below `Lower` with the `Lower` value.

---

## 🔴 Advanced Exercises

### Exercise 12.5: The KNN Imputer
**Objective**: Use Machine Learning algorithms to impute missing data intelligently!

**Instructions**:
We have a DataFrame with three columns: `['Age', 'ExperienceYears', 'Salary']`. There are `NaN` values in the `'Salary'` column.
Instead of replacing missing salaries with the average, write a snippet using `KNNImputer` from `sklearn.impute`.
1. Set `n_neighbors=5`.
2. Fit and transform the dataset. This will force the imputer to look at the 5 closest people (based on Age and Experience) and average their salaries to fill in the blank!

---

### Exercise 12.6: Refining Optimal Presentations
**Objective**: Turn a standard Plotly/Matplotlib plot into a boardroom-ready presentation.

**Instructions**:
Given a basic `sns.barplot(data=df, x='Department', y='Budget')`, write the specific `matplotlib.pyplot` methods required to:
1. Move the plot title, make it bold, and increase font size to `16`.
2. Remove the top and right axis spines (`sns.despine()`).
3. Add a grid, but ONLY the horizontal lines (`Y` axis) at `alpha=0.3`.
4. Rotate the `x-ticks` by 45 degrees so the department names don't overlap.

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`) or Jupyter Notebooks (`.ipynb`).
- Ensure all plots render correctly when the script is run.
- Name your files `exercise_12_X.py`.
