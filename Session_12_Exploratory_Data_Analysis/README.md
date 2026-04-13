# Session 12: Exploratory Data Analysis (EDA)

Welcome to **Session 12** of the Applied Machine Learning Using Python curriculum!

Before we ever train a Machine Learning model, we must fundamentally understand the data we are feeding into it. **"Garbage In, Garbage Out"** is the golden rule of AI. If you feed a state-of-the-art Neural Network data that is full of errors, missing values, or false patterns, the model will fail entirely.

Exploratory Data Analysis (EDA) is the art and science of investigating datasets to discover patterns, spot anomalies, test hypotheses, and verify assumptions using summary statistics and graphical representations.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Explain the role of EDA in identifying underlying patterns and trends in data.
2. List the importance of detecting irregularities (outliers).
3. Apply programmatic strategies for addressing missing values.
4. Refine data for optimal visual presentation.

---

## 📖 Part 1: Identifying Patterns and Trends

The primary goal of EDA is to uncover hidden relationships between variables (features) and the target we are trying to predict.

### Why is this important?
- **Feature Selection:** If we discover that "Customer Age" has zero correlation with "Will Default on Loan", we can drop the feature, saving computation time and reducing noise.
- **Multicollinearity:** If we find two features (e.g., "House Square Footage" and "House Number of Rooms") are highly correlated with *each other*, they provide redundant information. This can confuse linear models!
- **Data Distribution:** Does our data follow a normal (Gaussian) curve, or is it heavily skewed? Algorithms like Linear Regression assume normally distributed data!

**Common Tools:**
- **Correlation Heatmaps:** Visually map how strongly every feature correlates with every other feature.
- **Scatter Plots:** To identify linear vs. non-linear relationships.
- **Histograms & KDEs:** To visualize distributions.

---

## 📖 Part 2: Addressing Missing Values

Real-world datasets are never perfect. Web forms crash, sensors lose power, and humans leave boxes blank. 

### Why must we fix them?
Most Machine Learning algorithms (like standard Random Forests or Support Vector Machines) mathematically cannot perform calculations if there is a `NaN` (Not a Number) value in the array. They will simply crash.

### Strategies for Missing Data
1. **Deletion (`dropna`)**: The simplest method. If only 1% of your rows have missing data, just delete those rows. *Warning: If 40% is missing, deleting it throws away too much valuable information!*
2. **Mean/Median Imputation (`SimpleImputer`)**: Replace missing numerical values with the average or median of that column. Easy and fast, but reduces the variance of your data.
3. **Mode Imputation**: Replace missing categorical data with the most frequent class.
4. **Algorithmic Imputation (`KNNImputer`)**: Use K-Nearest Neighbors to guess what the missing value should be based on similar rows! Extremely powerful.

---

## 📖 Part 3: Detecting Irregularities (Outliers)

An **outlier** is an observation that lies an abnormal distance from other values in a random sample.
- *Example:* A dataset of human ages where one entry is `999` years old (likely a data entry error).
- *Example:* A dataset of daily temperatures where one day is -40°C in Summer (sensor malfunction).

### Why do they matter?
Outliers heavily distort the *Mean* and standard deviation. Models that rely on these metrics (like Linear Regression or K-Means Clustering) will be pulled toward the extreme outlier, ruining the predictions for the normal data.

### Detection Methods
1. **Z-Score:** Mathematically calculates how many standard deviations a value is away from the mean. Usually, a Z-score > 3 or < -3 is considered an outlier.
2. **IQR (Interquartile Range):** Defines bounds based on the 25th and 75th percentiles. Perfect for skewed data.
3. **Visual Detection:** Boxplots naturally highlight outliers as isolated dots beyond the "whiskers".

---

## 📖 Part 4: Refining Data for Optimal Presentation

The final step of EDA is refining your findings. When presenting ML insights to non-technical stakeholders (like a CEO), you cannot show them an array of raw floats. 
- Use **Seaborn** to create visually beautiful, color-blind friendly palettes.
- Add clear titles, X/Y axes labels, and contextual annotations.
- Group data logically (e.g., binning ages into "Teen", "Adult", "Senior" for cleaner bar charts).

---

## 🚀 Hands-On: Session Code Files

In the `code/` directory, you will find scripts demonstrating the concepts:
1. **`01_missing_values_and_outliers.py`**: Programmatically cleans a messy DataFrame using Z-Scores and KNN Imputers.
2. **`02_visualizing_patterns.py`**: Generates a professional Seaborn Correlation Heatmap and Scatterplots.

---
*© 2024 Aptech Limited — For Educational Use*
