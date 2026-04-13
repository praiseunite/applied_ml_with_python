# Session 13: Exploratory Data Analysis (Part 2)

Welcome to **Session 13** of the Applied Machine Learning Using Python curriculum!

In Part 1, we learned how to clean datasets, drop missing values, and visualize basic patterns. In Part 2, we transition to **Advanced EDA**. 

In Applied Machine Learning, we don't need to manually calculate complex mathematical formulas. Python libraries like `pandas` and `seaborn` have built-in functions designed to instantly provide deep statistical insights and multi-variate visualizations with a single line of code.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Identify the significance of **measures of central tendency and dispersion** using standard Pandas functions.
2. Explain the significance of **Advanced EDA Techniques** to extract deeper insights without relying purely on math.
3. Utilize advanced `Seaborn` matrices and pairplots for multivariate analysis.
4. Generate automated, comprehensive HTML data profiles using modern Python libraries.

---

## 📖 Part 1: Central Tendency and Dispersion (The Python Way)

When exploring a large dataset, we need numerical summaries to understand its "shape" before feeding it to an ML model.

### Central Tendency (Where is the middle?)
- **Mean (`df.mean()`):** The exact average. Great for perfectly symmetric data.
- **Median (`df.median()`):** The exact middle value. Essential for skewed data (e.g., if Bill Gates enters a room, the average income rockets to a billion dollars, but the median stays normal!).
- **Mode (`df.mode()`):** The most frequent value, used mostly for categorization (Categorical Data).

### Dispersion (How spread out is it?)
- **Standard Deviation (`df.std()`):** How far away, on average, data points are from the mean.
- **Range (`df.max() - df.min()`):** The absolute boundaries of your data.
- **Skewness (`df.skew()`):** Tells you if your data is leaning left or right. Algorithms like Linear Regression perform badly on highly skewed data!
- **Kurtosis (`df.kurtosis()`):** Tells you if your data has "heavy tails" (meaning lots of extreme outliers are present).

*In Python, you don't need to memorize these formulas! Running `df.describe()` instantly outputs primary central tendency and dispersion metrics!*

---

## 📖 Part 2: Advanced EDA (Multi-Variate Analysis)

Basic EDA looks at variables 1 or 2 at a time (e.g., a simple scatterplot). Advanced EDA looks at complex relationships across 3, 4, or dozens of features simultaneously.

### The Pairplot (`sns.pairplot`)
When you are handed a dataset with 8 different numerical columns, plotting every single possible combination is tedious. 
The Seaborn Pairplot automatically generates a massive grid, plotting every variable against every other variable instantly. This allows a Machine Learning Engineer to visually scan the entire dataset in 10 seconds to look for diagonal patterns or distinct clusters.

### The FacetGrid / JointPlot
- **FacetGrid:** Allows you to break down a single plot into a grid of multiple subplots based on a specific category (e.g., showing separate histograms for different countries side-by-side).
- **JointPlot:** Combines a scatter plot with histograms on the top and right edges, giving you both the relationship between two variables *and* their individual dispersions perfectly.

---

## 📖 Part 3: Automated EDA Profiling

While writing your own visualization scripts is essential for customization, industry professionals rely on **Automated Data Profiling** to speed up their workflow.

Libraries like `ydata-profiling` (formerly Pandas Profiling) can take any Raw Dataframe and automatically generate a polished, interactive HTML report containing:
- Warning sections for high cardinality, missing values, and high correlation.
- Interactive visualizations of every single variable.
- Full statistical dispersion analysis without writing a single line of matplotlib code.

Automated tools dramatically accelerate the ML pipeline!

---

## 🚀 Hands-On: Session Code Files

In the `code/` directory, you will find scripts demonstrating these Applied Python concepts:
1. **`01_central_tendency_python.py`**: Demonstrates the danger of relying on averages by comparing heavily skewed data using Pandas' built in metric methods (`.skew()`, `.median()`).
2. **`02_advanced_multivariate_plots.py`**: Generates complex, multi-dimensional `Seaborn` grids to extract deeper insights rapidly.

---
*© 2024 Aptech Limited — For Educational Use*
