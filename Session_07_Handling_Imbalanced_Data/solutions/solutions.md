# Solutions: Handling Imbalanced Data & PCA

---

## 🟢 BEGINNER: Feature Selection

**Answer 1:**
- **Filter Method:** Uses pure statistics (like correlations) to select features without ever running the Machine Learning algorithm. It is computationally extremely fast.
- **Wrapper Method:** Literally trains the ML algorithm over and over again, dropping one feature at a time to see what combination yields the best accuracy. It is highly accurate but computationally extremely slow.

**Answer 2:**
The accuracy will be 99.999% (99,999 healthy / 100,000 patients). This is a massive problem because the model completely failed its real-world objective: catching the 1 diseased patient. In this scenario, metric evaluation must switch to Recall and F1-scores.

---

## 🟡 INTERMEDIATE: PCA & Imbalanced Sets

**Answer 3:**
No, you will absolutely lose human interpretability. PCA mathematically "crushes" all 500 columns together to create 5 brand new "Principal Components." Component #1 is a mathematically weighted hybrid of all 500 original features, so it no longer represents "Age" or "Salary" by itself.

**Answer 4:**
Under-sampling forces you to throw away pieces of the majority class to balance the scales. If you have 1,000,000 normal transactions and use under-sampling, you will randomly delete 999,500 rows of valid data so that your data is limited to a 500 to 500 ratio! Your model will be completely starved of the rich context that existed in the deleted normal transactions.

---

## 🔴 ADVANCED: SMOTE Application

**Answer 5:**
Applying SMOTE *before* splitting the data causes massive **Data Leakage**. 
If you oversample the entire dataset, SMOTE generates synthetic copies that look highly similar to the original minority data. When you do the train/test split, highly similar data points will end up in *both* the training set and the testing set. The model will essentially memorize the synthetic fraud in the training set and then see identical variations of it in the test set, leading to 99% scores. When deployed in the real world, it will fail.
**Rule:** Always Split first, then apply SMOTE *only* to the Training Set.

**Answer 6:**
No, SMOTE does not just duplicate existing rows (that is mere random over-sampling). SMOTE uses k-Nearest Neighbors. It finds a real minority data point, draws a straight line to one of its nearest neighbor minority points, and randomly creates a brand new synthetic point mathematically seated on that line. It genuinely hallucinates brand-new, realistic data.
