# Session 19 — Exercises

## 🟢 Beginner Exercises

### Exercise 19.1: XAI Concepts — Transparency vs Interpretability
**Objective**: Distinguish between the three pillars of Explainable AI.

**Instructions**:
A hospital deploys a deep neural network to predict patient readmission risk. A patient is denied insurance coverage based on the model's prediction. The patient's lawyer demands to know:
1. **What** the model's internal architecture looks like (number of layers, activation functions).
2. **Why** this specific patient received a "High Risk" score.
3. **Who** is legally responsible for the decision.

For each of the lawyer's three demands, write a Python `print()` statement identifying which XAI pillar it corresponds to (Transparency, Interpretability, or Accountability) and briefly explain why.

### Exercise 19.2: Identifying Federated Learning Scenarios
**Objective**: Determine when Federated Learning is and isn't appropriate.

**Instructions**:
For each of the following scenarios, write a Python script that prints whether Federated Learning is **Required**, **Recommended**, or **Unnecessary**, along with a one-sentence justification:

1. A single research lab training a model on its own publicly available dataset.
2. Five competing banks building a shared fraud detection model.
3. A social media company training a content recommendation model on user data stored in one central data center.
4. Ten hospitals in different countries building a cancer detection model from patient MRIs.
5. A smartphone keyboard wanting to improve next-word prediction using how each user types.

---

## 🟡 Intermediate Exercises

### Exercise 19.3: Building a Federated Averaging Function
**Objective**: Implement the core FedAvg aggregation algorithm from scratch.

**Instructions**:
Write a Python function `federated_average(client_updates)` that:
1. Accepts a list of dictionaries, where each dictionary contains:
   - `'weights'`: a numpy array of model weights
   - `'data_size'`: an integer representing how many samples this client trained on
2. Computes the **weighted average** of all weight vectors (weighted by `data_size`).
3. Returns the aggregated global weight vector.

Test your function with the following mock data:
```python
client_updates = [
    {'weights': np.array([1.0, 2.0, 3.0]), 'data_size': 100},
    {'weights': np.array([2.0, 3.0, 4.0]), 'data_size': 200},
    {'weights': np.array([0.5, 1.5, 2.5]), 'data_size': 50},
]
```
Print the result and verify it manually.

### Exercise 19.4: TF-IDF Document Ranking
**Objective**: Build a minimal document ranker from scratch.

**Instructions**:
Given the following 4 mini-documents:
```python
docs = [
    "Machine learning models require large datasets for training",
    "Deep learning is a subset of machine learning using neural networks",
    "Natural language processing enables computers to understand human text",
    "Reinforcement learning teaches agents through reward and punishment"
]
```

1. Use `TfidfVectorizer` from scikit-learn to vectorize all 4 documents.
2. Write a query: `"How do neural networks learn from data?"`
3. Compute the cosine similarity between the query and each document.
4. Print the documents ranked by relevance score (highest first).
5. Print which document was selected as #1 and explain in a comment why it makes sense.

---

## 🔴 Advanced Exercises

### Exercise 19.5: Differential Privacy Simulation
**Objective**: Understand how adding noise protects privacy in Federated Learning.

**Instructions**:
A malicious Federated Learning server tries to reverse-engineer a client's data from its gradient update. Simulate this:

1. Create a "true gradient" vector: `np.array([0.5, -0.3, 0.8, -0.1, 0.6])`
2. Write a function `add_differential_privacy(gradient, epsilon)` that adds Laplace noise to each element. The noise should be drawn from `np.random.laplace(0, 1/epsilon, len(gradient))`.
3. Test with `epsilon = 0.1` (High Privacy), `epsilon = 1.0` (Medium Privacy), and `epsilon = 10.0` (Low Privacy).
4. Print the noisy gradient at each privacy level alongside the original.
5. Compute and print the Mean Absolute Error (MAE) between the original and noisy gradients at each level.
6. In a comment, explain the privacy-utility trade-off: why does higher privacy (smaller epsilon) make the model less accurate?

### Exercise 19.6: Edge AI Model Compression
**Objective**: Demonstrate why Edge AI requires model compression.

**Instructions**:
1. Train a `RandomForestClassifier` with `n_estimators=500` on the Iris dataset.
2. Use Python's `pickle` module to serialize the model and measure its file size in KB.
3. Train a second `RandomForestClassifier` with `n_estimators=10` (simulating an "Edge-optimized" model).
4. Serialize it and measure its size.
5. Evaluate both models' accuracy on a test split.
6. Print a comparison table showing: Model Name, Accuracy, File Size (KB), and compute the "Accuracy per KB" ratio.
7. In a comment, explain why the smaller model is more appropriate for deployment on a drone or smartphone.

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`).
- Answer conceptual questions using `print()` statements in your scripts.
- Name your files `exercise_19_X.py`.
