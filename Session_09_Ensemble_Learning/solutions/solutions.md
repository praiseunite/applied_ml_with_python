# Solutions: Ensemble Learning & Evaluation

---

## 🟢 BEGINNER: Core Concepts

**Answer 1:**
- **Hard Voting** takes the simple majority vote. If two models say Class A and one says Class B, it outputs Class A.
- **Soft Voting** averages the predicted probabilities from each algorithm. If Model 1 says 90% Class A, Model 2 says 51% Class B, and Model 3 says 55% Class B, Soft voting will yield Class A because the average confidence for A is much higher.

**Answer 2:**
Parallel. Random Forest is a Bagging algorithm, meaning all the trees are completely independent of each other and can be trained simultaneously across different CPU cores.

**Answer 3:**
The accuracy will be 99%, making the model look fantastic. However, it is completely useless because it failed to catch the 1% of actual fraud. Accuracy is a terrible metric for imbalanced data. Instead, metrics like Precision, Recall, or F1-Score should be used.

---

## 🟡 INTERMEDIATE: Mechanics & implementation

**Answer 4:**
Bootstrapping means selecting random samples from the dataset *with replacement*. It is necessary because if every Decision Tree was trained on the exact same dataset, they would all make the exact same splits, defeating the purpose of an ensemble mathematically.

**Answer 5:**
The learning rate scales the contribution of each newly added tree. A lower learning rate means each tree makes smaller corrections to the overall prediction. If the learning rate is too high, the model will aggressively overcorrect errors, causing wild oscillations and severely overfitting the training data.

**Answer 6:**
An ROC-AUC score of 0.51 means the model is essentially guessing randomly (a coin flip gives an AUC of 0.50). The model has failed to learn any meaningful patterns distinguishing the positive class from the negative class.

---

## 🔴 ADVANCED: Hyperparameters & Optimization

**Answer 7:**
Because Random Forests train trees independently, taking the average of 10,000 independent trees just continues to reduce variance (though diminishing returns apply). In contrast, Gradient Boosting trains sequentially, where each new tree specifically targets the residual errors of the previous trees. If you add too many trees, the model will eventually start learning the random noise in the training data perfectly, resulting in severe overfitting and decreased performance on unseen testing data.

**Answer 8:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(random_state=42)

# cv=5 indicates 5-fold cross validation. 
# scoring='roc_auc' tells it to evaluate using Area Under the ROC Curve.
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')

print(f"Mean CV ROC-AUC: {scores.mean():.4f}")
```
