# Session 19 — Solutions

## 🟢 Solution 19.1: XAI Concepts — Transparency vs Interpretability

```python
print("Demand 1: 'Show me the model architecture (layers, activations).'")
print("→ This is TRANSPARENCY. It asks whether the model's internal structure can be inspected.")
print("  A Decision Tree is transparent. A 50-layer neural network is opaque.\n")

print("Demand 2: 'Explain WHY this specific patient got a High Risk score.'")
print("→ This is INTERPRETABILITY. It asks for a causal explanation of a specific prediction.")
print("  Tools like LIME and SHAP provide per-prediction interpretability.\n")

print("Demand 3: 'Who is legally responsible for this decision?'")
print("→ This is ACCOUNTABILITY. It asks who bears the legal and ethical consequences.")
print("  Under the EU AI Act, the deployer of a high-risk AI must provide explanations on demand.")
```

## 🟢 Solution 19.2: Identifying Federated Learning Scenarios

```python
scenarios = [
    ("Single lab, own public data", "Unnecessary",
     "No privacy concern — the data is public and already centralized."),
    ("Five competing banks, fraud detection", "Required",
     "Banks cannot legally share customer financial records. FL lets them collaborate without exposure."),
    ("Social media, centralized data", "Unnecessary",
     "Data is already in one place. Standard centralized training is simpler and more efficient."),
    ("Ten hospitals, cancer MRIs", "Required",
     "Patient medical data is protected by HIPAA/GDPR. FL enables collaboration without data movement."),
    ("Smartphone keyboard, user typing", "Required",
     "User keystrokes are highly personal. Google pioneered FL specifically for this use case (Gboard)."),
]

for scenario, verdict, reason in scenarios:
    print(f"Scenario: {scenario}")
    print(f"  Verdict: {verdict}")
    print(f"  Reason:  {reason}\n")
```

---

## 🟡 Solution 19.3: Building a Federated Averaging Function

```python
import numpy as np

def federated_average(client_updates):
    total_samples = sum(u['data_size'] for u in client_updates)
    weighted_sum = sum(
        u['weights'] * (u['data_size'] / total_samples)
        for u in client_updates
    )
    return weighted_sum

# Test
client_updates = [
    {'weights': np.array([1.0, 2.0, 3.0]), 'data_size': 100},
    {'weights': np.array([2.0, 3.0, 4.0]), 'data_size': 200},
    {'weights': np.array([0.5, 1.5, 2.5]), 'data_size': 50},
]

result = federated_average(client_updates)
print(f"Aggregated Global Weights: {result}")
# Expected: weighted avg = (100*[1,2,3] + 200*[2,3,4] + 50*[0.5,1.5,2.5]) / 350
# = ([100,200,300] + [400,600,800] + [25,75,125]) / 350
# = [525, 875, 1225] / 350
# = [1.5, 2.5, 3.5]
print(f"Expected: [1.5, 2.5, 3.5]")
```

## 🟡 Solution 19.4: TF-IDF Document Ranking

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "Machine learning models require large datasets for training",
    "Deep learning is a subset of machine learning using neural networks",
    "Natural language processing enables computers to understand human text",
    "Reinforcement learning teaches agents through reward and punishment"
]

query = "How do neural networks learn from data?"

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(docs)
query_vec = vectorizer.transform([query])

similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
ranked = similarities.argsort()[::-1]

for rank, idx in enumerate(ranked, 1):
    print(f"  #{rank}: [{similarities[idx]:.3f}] {docs[idx]}")

# Document #2 should rank first because it shares the most keywords:
# "neural networks", "learning", and "machine learning" all appear
# in both the query and document #2.
```

---

## 🔴 Solution 19.5: Differential Privacy Simulation

```python
import numpy as np

def add_differential_privacy(gradient, epsilon):
    noise = np.random.laplace(0, 1/epsilon, len(gradient))
    return gradient + noise

np.random.seed(42)
true_gradient = np.array([0.5, -0.3, 0.8, -0.1, 0.6])

for epsilon, label in [(0.1, "High Privacy"), (1.0, "Medium Privacy"), (10.0, "Low Privacy")]:
    noisy = add_differential_privacy(true_gradient.copy(), epsilon)
    mae = np.mean(np.abs(true_gradient - noisy))
    print(f"[ε={epsilon:>5.1f}] {label:>15s} | Noisy: {np.round(noisy, 3)} | MAE: {mae:.4f}")

# Privacy-Utility Trade-off:
# Smaller epsilon → more noise → attacker cannot recover original data → model trains on noisier gradients → lower accuracy.
# Larger epsilon → less noise → faster convergence → but attacker can more easily reconstruct the original data.
# The art is choosing an epsilon that balances privacy guarantees with acceptable model performance degradation.
```

## 🔴 Solution 19.6: Edge AI Model Compression

```python
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Full model
full_model = RandomForestClassifier(n_estimators=500, random_state=42)
full_model.fit(X_train, y_train)
full_acc = accuracy_score(y_test, full_model.predict(X_test))
full_bytes = len(pickle.dumps(full_model))

# Edge model
edge_model = RandomForestClassifier(n_estimators=10, random_state=42)
edge_model.fit(X_train, y_train)
edge_acc = accuracy_score(y_test, edge_model.predict(X_test))
edge_bytes = len(pickle.dumps(edge_model))

print(f"{'Model':<20} {'Accuracy':>10} {'Size (KB)':>10} {'Acc/KB':>10}")
print("-" * 52)
print(f"{'Full (500 trees)':<20} {full_acc*100:>9.1f}% {full_bytes/1024:>9.1f} {full_acc/(full_bytes/1024):>10.4f}")
print(f"{'Edge (10 trees)':<20} {edge_acc*100:>9.1f}% {edge_bytes/1024:>9.1f} {edge_acc/(edge_bytes/1024):>10.4f}")

# The Edge model is ~50x smaller with minimal accuracy loss.
# On a drone with 256KB of RAM, only the edge model fits.
# The accuracy-per-KB ratio proves the edge model is vastly more efficient.
```
