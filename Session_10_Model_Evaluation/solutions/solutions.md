# Solutions: Model Evaluation

---

## 🟢 BEGINNER: Bias vs Variance

**Answer 1:**
It is suffering from **High Variance (Overfitting)**. The massive gap between the Training score (99.8%) and the Testing score (45%) proves that the model perfectly memorized the specific spam emails in the training database, but completely failed to learn generalized rules about what makes an email spam in the real world.

**Answer 2:**
It is **Underfitting (High Bias)**. Because the accuracy is uniformly terrible on both the training and testing sets, it means the algorithmic model is simply too basic (a straight line) to capture the complexity of the data (curved housing prices).

---

## 🟡 INTERMEDIATE: Cross-Validation Types

**Answer 3:**
Stratified K-Fold ensures that the chronological percentage of classes remains exactly identical across every single fold. For example, if your overall dataset is 90% Class A and 10% Class B, Stratified guarantees that every single pop-quiz (Fold) is exactly 90/10. It is absolutely mandatory for highly imbalanced datasets like Credit Card Fraud or Rare Disease detection.

**Answer 4:**
You will have to train the model **5,000 separate times**. In LOOCV, the model trains on 4,999 rows and tests on the 1 row left out. The main drawback is that it is incredibly computationally expensive and literally impossible for complex models like Neural Networks on large datasets.

---

## 🔴 ADVANCED: ROC Curve Mechanics

**Answer 5:**
An AUC of 0.30 implies the model is predicting *worse* than random guessing. It actually means the model has successfully found the mathematical pattern, but gets the answer completely backward (it predicts Fraud when it's Normal, and Normal when it's Fraud). You can "fix" this by simply inverting the probabilities (e.g., $1 - P_{score}$), which instantly turns your 0.30 failure model into a highly successful 0.70 model!

**Answer 6:**
The perfect point is located at the absolute **Top-Left Corner** of the graph (Coordinate `0, 1`). 
- The X-axis represents the False Positive Rate (we want this at `0`).
- The Y-axis represents the True Positive Rate (we want this at `1`).
