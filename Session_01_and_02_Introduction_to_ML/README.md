# Session 01 & 02: Introduction to Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Duration-4%20Hours-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TL1%20+%20TL2-Sessions%201--2-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Level-All%20Levels-green?style=for-the-badge" />
</p>

> **Covers**: TL1 (Session 1) and TL2 (Session 2) — 4 hours total
> 
> **Book Reference**: Applied Machine Learning Using Python — Session 1

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:

| # | Objective | Level |
|---|-----------|-------|
| 1 | Explain the purpose of ML and its historical significance | 🟢 Beginner |
| 2 | List ethical considerations in AI with real-world examples | 🟢 Beginner |
| 3 | Explain why ML is used in data processing and identify real-world applications | 🟢 Beginner |
| 4 | Organize bias and fairness in ML | 🟡 Intermediate |
| 5 | Assess real-world ethical dilemmas in AI | 🟡 Intermediate |
| 6 | Summarize ML's impact on data processing and ethical implications | 🔴 Advanced |

---

## 📋 Prerequisites

- Basic Python programming (variables, functions, loops, classes)
- Familiarity with NumPy and pandas
- Basic understanding of statistics (mean, median, standard deviation)

---

## Table of Contents

- [Part 1: What is Machine Learning?](#part-1-what-is-machine-learning)
- [Part 2: History of Machine Learning](#part-2-history-of-machine-learning)
- [Part 3: Types of Machine Learning](#part-3-types-of-machine-learning)
- [Part 4: The ML Pipeline](#part-4-the-ml-pipeline)
- [Part 5: Real-World Applications](#part-5-real-world-applications-of-ml)
- [Part 6: Ethics, Bias, and Fairness in AI](#part-6-ethics-bias-and-fairness-in-ai)
- [Part 7: Responsible AI Practices](#part-7-responsible-ai-practices)
- [Hands-On Lab](#-hands-on-lab)
- [Portfolio Task](#-portfolio-task)
- [Exercises](#-exercises)

---

## Part 1: What is Machine Learning?

### 🟢 Beginner Level

**Machine Learning (ML)** is a subset of Artificial Intelligence (AI) that enables computers to learn patterns from data and make decisions **without being explicitly programmed** for every scenario.

#### The Key Idea

Think of it this way:

| Traditional Programming | Machine Learning |
|------------------------|------------------|
| Input: **Data + Rules** | Input: **Data + Answers** |
| Output: **Answers** | Output: **Rules** |

**Traditional Programming Example:**
```python
# Spam filter — traditional approach
# A human writes ALL the rules
def is_spam(email):
    if "buy now" in email.lower():
        return True
    if "free money" in email.lower():
        return True
    if "click here" in email.lower():
        return True
    # ... hundreds more rules ...
    return False
```

**Machine Learning Approach:**
```python
# Spam filter — ML approach
# The algorithm LEARNS the rules from data
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Data: thousands of emails labeled as "spam" or "not spam"
emails = ["Buy now! Free money!", "Meeting at 3pm", "Click here to win!", ...]
labels = ["spam", "not_spam", "spam", ...]

# The model discovers its own rules
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

# Now it can classify NEW emails it has never seen
new_email = vectorizer.transform(["Congratulations! You won a prize!"])
prediction = model.predict(new_email)  # → "spam"
```

> **💡 Why is this powerful?** Spammers constantly change their tactics. A rule-based system needs a human to update rules. An ML model can be retrained on new data and adapt automatically.

#### Formal Definition

> **Machine Learning** is the field of study that gives computers the ability to learn without being explicitly programmed. — *Arthur Samuel, 1959*

More precisely:

> A computer program is said to **learn** from experience **E** with respect to some task **T** and performance measure **P**, if its performance at **T**, as measured by **P**, improves with experience **E**. — *Tom Mitchell, 1997*

**Example using Mitchell's definition:**

| Component | Email Spam Filter |
|-----------|------------------|
| **Task (T)** | Classify emails as spam or not spam |
| **Experience (E)** | A dataset of labeled emails |
| **Performance (P)** | Accuracy — % of correctly classified emails |

---

### 🟡 Intermediate Level

#### AI vs ML vs Deep Learning vs Data Science

Understanding where ML fits in the broader landscape:

```
┌──────────────────────────────────────────┐
│           Artificial Intelligence         │
│   "Machines that can perform tasks        │
│    that typically require human           │
│    intelligence"                          │
│                                          │
│   ┌──────────────────────────────────┐   │
│   │       Machine Learning           │   │
│   │   "Algorithms that learn from    │   │
│   │    data"                         │   │
│   │                                  │   │
│   │   ┌──────────────────────────┐   │   │
│   │   │     Deep Learning        │   │   │
│   │   │   "Neural networks with  │   │   │
│   │   │    multiple layers"      │   │   │
│   │   └──────────────────────────┘   │   │
│   └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

| Field | Focus | Example |
|-------|-------|---------|
| **AI** | Systems that mimic human intelligence | Siri, self-driving cars |
| **ML** | Algorithms that learn patterns from data | Recommendation engines, fraud detection |
| **Deep Learning** | Neural networks with many layers for complex patterns | Image recognition, language translation |
| **Data Science** | Extracting insights from data using statistics + ML + domain knowledge | Business analytics, A/B testing |

#### When to Use ML (and When NOT To)

| ✅ Use ML When | ❌ Don't Use ML When |
|----------------|---------------------|
| Complex patterns exist in data | Simple rules can solve the problem |
| You have enough quality data | You have very little data |
| The problem is well-defined | The problem is vague or undefined |
| Patterns change over time | Rules are fixed and well-known |
| Human expertise is hard to encode | A lookup table or formula works |

---

### 🔴 Advanced Level

#### The Computational Learning Theory Perspective

Machine Learning has formal mathematical foundations rooted in **computational learning theory**:

- **PAC Learning (Probably Approximately Correct)**: Proposed by Leslie Valiant (1984). A concept class is PAC-learnable if there exists an algorithm that, with high probability (≥ 1-δ), produces a hypothesis with low error (≤ ε), in polynomial time.

- **VC Dimension (Vapnik-Chervonenkis)**: Measures the capacity of a model class. A set of points is **shattered** by a model if the model can perfectly separate them for every possible labeling. The VC dimension is the largest set size that can be shattered.

  - Linear classifier in 2D: VC dimension = 3 (can shatter 3 points, not 4)
  - This connects to the **bias-variance tradeoff**: models with higher VC dimensions can fit more complex patterns but risk overfitting.

- **No Free Lunch Theorem (Wolpert & Macready, 1997)**: No single algorithm performs best on every problem. This is why we study multiple algorithms throughout this course.

**Reference**: Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.

---

## Part 2: History of Machine Learning

### 🟢 Beginner Level

A timeline of key milestones:

| Year | Milestone | Significance |
|------|-----------|-------------|
| **1943** | McCulloch & Pitts | First mathematical model of a neuron |
| **1950** | Alan Turing's "Computing Machinery and Intelligence" | Proposed the Turing Test — can machines think? |
| **1957** | Frank Rosenblatt's Perceptron | First neural network — could learn simple patterns |
| **1959** | Arthur Samuel | Coined the term "Machine Learning" (checkers program) |
| **1967** | Nearest Neighbor algorithm | First practical classification algorithm |
| **1979** | Stanford Cart | One of the first autonomous vehicles |
| **1986** | Backpropagation (Rumelhart, Hinton, Williams) | Made training deep networks possible |
| **1997** | IBM Deep Blue beats Kasparov | AI beats world chess champion |
| **2006** | Geoffrey Hinton — Deep Learning renaissance | Deep Belief Networks reignited neural network research |
| **2011** | IBM Watson wins Jeopardy! | NLP milestone — AI understanding natural language |
| **2012** | AlexNet wins ImageNet | Deep Learning revolution begins — CNNs dominate computer vision |
| **2014** | GANs (Goodfellow et al.) | Generative Adversarial Networks — AI creates realistic images |
| **2016** | AlphaGo beats Lee Sedol | DeepMind's RL agent masters the game of Go |
| **2017** | Transformer architecture (Vaswani et al.) | "Attention Is All You Need" — foundation for modern LLMs |
| **2020** | GPT-3 (OpenAI) | 175B parameter language model |
| **2022** | ChatGPT launch | Generative AI goes mainstream |
| **2023** | GPT-4, Gemini, Claude | Multi-modal AI, rapid industry adoption |

### 🟡 Intermediate Level — The AI Winters

ML progress wasn't linear. The field experienced two major **"AI Winters"** — periods of reduced funding and disappointment:

| Period | AI Winter | Cause | What Ended It |
|--------|-----------|-------|---------------|
| 1974–1980 | First AI Winter | Perceptron limitations (Minsky & Papert), overpromised results | Expert systems, Japanese 5th Gen Computing |
| 1987–1993 | Second AI Winter | Expert systems failed to scale, expensive hardware | Internet data explosion, faster GPUs, SVM & ensemble methods |

> **Key lesson**: ML requires three things to succeed: **(1) algorithms, (2) compute power, (3) data**. In the early years, we had (1) but not (2) or (3). The modern ML revolution happened because all three converged.

---

## Part 3: Types of Machine Learning

### 🟢 Beginner Level

Machine Learning is divided into three main categories:

```
                    Machine Learning
                    ┌──────┼──────┐
                    │      │      │
              Supervised  Unsupervised  Reinforcement
              Learning    Learning      Learning
              │           │             │
              ├─ Classification  ├─ Clustering    ├─ Policy-based
              ├─ Regression      ├─ Dimensionality├─ Value-based
              └─ Ranking         │  Reduction     └─ Model-based
                                 └─ Association
```

#### 1. Supervised Learning

The model learns from **labeled data** — each input has a known correct output.

| Type | Goal | Output | Example |
|------|------|--------|---------|
| **Classification** | Predict a category | Discrete label | Email → Spam or Not Spam |
| **Regression** | Predict a number | Continuous value | House features → Price |

```python
# Classification Example: Predict flower species
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load labeled data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train model on labeled data
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
# Output: Accuracy: 97.78%
```

```python
# Regression Example: Predict house prices
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load labeled data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: ${rmse * 100000:.0f}")
# Output: RMSE: ~$74,000 (average prediction error)
```

#### 2. Unsupervised Learning

The model finds **hidden patterns** in data **without labels**.

| Type | Goal | Example |
|------|------|---------|
| **Clustering** | Group similar items | Customer segmentation |
| **Dimensionality Reduction** | Simplify data, keep important info | PCA on images |
| **Anomaly Detection** | Find unusual data points | Fraud detection |

```python
# Clustering Example: Group customers
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample customer data (no labels!)
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Find groups automatically
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', s=200, c='red', label='Centers')
plt.title("Customer Segments Discovered by K-Means")
plt.legend()
plt.show()
```

#### 3. Reinforcement Learning

The model learns by **trial and error**, receiving **rewards** for good actions and **penalties** for bad ones.

| Component | Description | Example |
|-----------|-------------|---------|
| **Agent** | The learner/decision-maker | A game-playing AI |
| **Environment** | The world the agent interacts with | The game itself |
| **Action** | What the agent can do | Move left, right, jump |
| **Reward** | Feedback signal | +1 for scoring, -1 for losing |
| **Policy** | The agent's strategy | "When I see X, do Y" |

```python
# Simple RL Concept: Epsilon-Greedy Exploration
import numpy as np

# Imagine a slot machine with 3 arms
n_arms = 3
true_rewards = [0.2, 0.5, 0.8]  # Arm 3 is best (unknown to agent)
estimated_rewards = [0.0, 0.0, 0.0]
arm_counts = [0, 0, 0]
epsilon = 0.1  # Explore 10% of the time

total_reward = 0
for step in range(1000):
    # Epsilon-greedy: explore or exploit?
    if np.random.random() < epsilon:
        arm = np.random.randint(n_arms)  # Explore: random arm
    else:
        arm = np.argmax(estimated_rewards)  # Exploit: best known arm
    
    # Get reward (simulated)
    reward = 1 if np.random.random() < true_rewards[arm] else 0
    total_reward += reward
    
    # Update estimate (incremental mean)
    arm_counts[arm] += 1
    estimated_rewards[arm] += (reward - estimated_rewards[arm]) / arm_counts[arm]

print(f"Estimated rewards: {[f'{r:.2f}' for r in estimated_rewards]}")
print(f"True rewards:      {true_rewards}")
print(f"Total reward:      {total_reward}/1000")
```

---

### 🟡 Intermediate Level

#### Semi-Supervised and Self-Supervised Learning

| Type | How It Works | Use Case |
|------|-------------|----------|
| **Semi-Supervised** | Small amount of labeled data + large amount of unlabeled data | Medical imaging (expensive to label) |
| **Self-Supervised** | Creates its own labels from data (e.g., predicting missing words) | GPT, BERT, large language models |
| **Transfer Learning** | Reuse a model trained on one task for a different task | Fine-tune ImageNet model for X-rays |

#### Algorithm Selection Guide

```
Start
│
├─ Do you have labeled data?
│  ├─ YES → Supervised Learning
│  │  ├─ Is the output a category?
│  │  │  ├─ YES → Classification
│  │  │  │  ├─ Linear decision boundary? → Logistic Regression, SVM
│  │  │  │  ├─ Non-linear? → Random Forest, XGBoost, Neural Net
│  │  │  │  └─ Need interpretability? → Decision Tree, Logistic Regression
│  │  │  └─ NO → Regression
│  │  │     ├─ Linear relationship? → Linear Regression, Ridge, Lasso
│  │  │     └─ Non-linear? → Random Forest, Gradient Boosting, SVR
│  │  └─ How much data do you have?
│  │     ├─ Small (< 1K) → SVM, KNN, simple models
│  │     ├─ Medium (1K–100K) → Random Forest, XGBoost
│  │     └─ Large (100K+) → Deep Learning, LightGBM
│  │
│  └─ NO → Unsupervised Learning
│     ├─ Find groups? → K-Means, DBSCAN, Hierarchical
│     ├─ Reduce dimensions? → PCA, t-SNE, UMAP
│     └─ Find anomalies? → Isolation Forest, LOF
│
└─ Is there a reward signal?
   └─ YES → Reinforcement Learning
      ├─ Discrete actions? → Q-Learning, DQN
      └─ Continuous actions? → Policy Gradient, Actor-Critic
```

---

### 🔴 Advanced Level

#### The Bias-Variance Tradeoff

Every ML model's error can be decomposed:

```
Total Error = Bias² + Variance + Irreducible Error
```

| Component | What It Means | High When |
|-----------|---------------|-----------|
| **Bias** | Error from oversimplifying the model | Model is too simple (underfitting) |
| **Variance** | Error from being too sensitive to training data | Model is too complex (overfitting) |
| **Irreducible Error** | Noise inherent in the data | Always present, can't be reduced |

```python
# Demonstrating Bias-Variance Tradeoff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Generate noisy sine wave data
np.random.seed(42)
X = np.sort(np.random.uniform(0, 1, 100)).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.2, 100)

# Try different model complexities
degrees = [1, 3, 5, 10, 15]
fig, axes = plt.subplots(1, len(degrees), figsize=(20, 4))

for ax, degree in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    cv_score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    
    ax.scatter(X, y, alpha=0.3, s=10)
    ax.plot(X_plot, y_plot, 'r-', linewidth=2)
    ax.set_title(f"Degree {degree}\nCV MSE: {-cv_score.mean():.3f}")
    ax.set_ylim(-2, 2)

plt.suptitle("Bias-Variance Tradeoff: Polynomial Regression", fontsize=14)
plt.tight_layout()
plt.show()
# Degree 1: High bias (underfitting)
# Degree 3-5: Good balance
# Degree 15: High variance (overfitting)
```

---

## Part 4: The ML Pipeline

### 🟢 Beginner Level

Every ML project follows this pipeline:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Problem  │───▶│  Data    │───▶│  Data    │───▶│  Model   │───▶│  Model   │───▶│  Deploy  │
│ Define   │    │ Collect  │    │ Prepare  │    │ Train    │    │ Evaluate │    │  Model   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

#### Complete Pipeline Example

```python
"""
Complete ML Pipeline: Predicting Diabetes Progression
Dataset: sklearn's diabetes dataset (real medical data)
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# Step 1: Problem Definition
# ──────────────────────────────────────────
# Goal: Predict disease progression one year after baseline
# This is a REGRESSION problem (predicting a continuous value)

# ──────────────────────────────────────────
# Step 2: Data Collection
# ──────────────────────────────────────────
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="progression")

print("Dataset Shape:", X.shape)
print("Features:", list(X.columns))
print(f"\nTarget range: {y.min():.0f} to {y.max():.0f}")
print(f"Target mean:  {y.mean():.0f}")
print("\nFirst 5 rows:")
print(X.head())

# ──────────────────────────────────────────
# Step 3: Data Preparation
# ──────────────────────────────────────────
# Check for missing values
print(f"\nMissing values: {X.isnull().sum().sum()}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# Scale features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────
# Step 4: Model Training (Compare Multiple Models)
# ──────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Lasso Regression":  Lasso(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation (more reliable estimate)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                 scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {"RMSE": rmse, "R²": r2, "CV_RMSE": cv_rmse}
    print(f"{name:25s} → RMSE: {rmse:.2f} | R²: {r2:.3f} | CV RMSE: {cv_rmse:.2f}")

# ──────────────────────────────────────────
# Step 5: Model Evaluation & Visualization
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model comparison
names = list(results.keys())
rmses = [results[n]["RMSE"] for n in names]
r2s = [results[n]["R²"] for n in names]

axes[0].barh(names, rmses, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
axes[0].set_xlabel("RMSE (Lower is Better)")
axes[0].set_title("Model Comparison: RMSE")

# Plot 2: Actual vs Predicted (best model)
best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test_scaled)
axes[1].scatter(y_test, y_pred_best, alpha=0.5, color='#3498db')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel("Actual Progression")
axes[1].set_ylabel("Predicted Progression")
axes[1].set_title("Random Forest: Actual vs Predicted")
axes[1].legend()

plt.tight_layout()
plt.savefig("ml_pipeline_results.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Pipeline complete! Results saved to ml_pipeline_results.png")
```

### 🟡 Intermediate Level — Scikit-Learn Pipeline API

```python
"""
Production-Grade Pipeline using sklearn.pipeline
This is how ML code should be written in industry.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# Use California Housing dataset (more realistic than diabetes)
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Automatic preprocessing + training pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, 
    scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=1
)

grid_search.fit(X, y)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
print(f"Best R² Score: {grid_search.best_estimator_.score(X, y):.4f}")
```

---

## Part 5: Real-World Applications of ML

### 🟢 Beginner Level

ML is everywhere in modern technology:

| Industry | Application | ML Type | Algorithm Used |
|----------|-------------|---------|----------------|
| **Healthcare** | Disease diagnosis from X-rays | Classification | CNN (Deep Learning) |
| **Finance** | Credit card fraud detection | Classification | Random Forest, XGBoost |
| **E-commerce** | Product recommendations | Recommendation | Collaborative Filtering |
| **Transportation** | Self-driving cars | RL + Deep Learning | CNN + RL |
| **Entertainment** | Netflix show recommendations | Recommendation | Matrix Factorization |
| **Agriculture** | Crop disease detection | Classification | Transfer Learning (CNN) |
| **Manufacturing** | Predictive maintenance | Anomaly Detection | Isolation Forest |
| **Marketing** | Customer segmentation | Clustering | K-Means, DBSCAN |
| **NLP** | Language translation | Seq2Seq | Transformer |
| **Cybersecurity** | Intrusion detection | Anomaly Detection | Autoencoders |

### 🟡 Intermediate Level — ML in Data Processing

Why ML is critical for modern data processing:

1. **Volume**: Datasets are too large for manual analysis (Netflix processes 1.5 PB/day)
2. **Velocity**: Real-time decisions needed (trading, fraud detection)
3. **Variety**: Unstructured data (text, images, audio) needs ML to extract insights
4. **Veracity**: ML can detect data quality issues and anomalies automatically

```python
"""
Real-World Example: Automated Data Quality Check with ML
Uses Isolation Forest to detect data anomalies
"""
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load real estate data
housing = fetch_california_housing(as_frame=True)
df = housing.data.copy()

# Inject some anomalies (simulating real-world data quality issues)
np.random.seed(42)
anomaly_indices = np.random.choice(len(df), 50, replace=False)
df.loc[anomaly_indices, 'MedInc'] = df['MedInc'].max() * 10  # Extreme income values

# Detect anomalies using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(df)

# Results
n_anomalies = (anomaly_labels == -1).sum()
detected = np.isin(np.where(anomaly_labels == -1)[0], anomaly_indices).sum()

print(f"Total records:        {len(df)}")
print(f"Anomalies detected:   {n_anomalies}")
print(f"True anomalies found: {detected} out of {len(anomaly_indices)} injected")
print(f"Detection rate:       {detected/len(anomaly_indices):.1%}")
```

---

## Part 6: Ethics, Bias, and Fairness in AI

> ⚠️ **This section is critical.** AI systems make decisions that affect people's lives — loans, hiring, criminal sentencing, healthcare. Understanding ethics is not optional.

### 🟢 Beginner Level

#### What is Bias in AI?

**Bias in ML** occurs when an algorithm's output systematically favors or disadvantages certain groups. This usually happens because the **training data** reflects existing societal biases.

#### Real-World Case Studies of AI Bias

| Case | What Happened | Impact | Root Cause |
|------|---------------|--------|------------|
| **Amazon Hiring Tool (2018)** | AI recruiting tool downgraded résumés with the word "women's" (e.g., "women's chess club") | Gender discrimination in hiring | Trained on 10 years of male-dominated hiring data |
| **COMPAS Recidivism (2016)** | Criminal sentencing algorithm was more likely to falsely label Black defendants as high risk | Racial discrimination in criminal justice | Biased historical arrest data |
| **Google Photos (2015)** | Image classifier labeled Black users as "gorillas" | Racial insensitivity | Insufficient diversity in training data |
| **Healthcare Algorithm (2019)** | Hospital algorithm used healthcare costs as a proxy for health needs, systematically underserving Black patients | Racial disparity in healthcare allocation | Biased proxy variable — Black patients had lower costs due to access barriers, not better health |
| **Apple Card (2019)** | Goldman Sachs' algorithm gave men higher credit limits than women with similar profiles | Gender discrimination in finance | Opaque algorithm with no bias testing |

### 🟡 Intermediate Level

#### Types of Bias in ML

```
Data Bias                    Algorithmic Bias              Human Bias
├─ Selection Bias            ├─ Evaluation Bias            ├─ Confirmation Bias
│  (Data doesn't represent   │  (Wrong metrics used)       │  (Seeking confirming
│   the real population)     │                             │   evidence)
├─ Measurement Bias          ├─ Aggregation Bias           ├─ Automation Bias
│  (Inconsistent data        │  (One model for all groups  │  (Over-trusting AI
│   collection)              │   when groups differ)       │   decisions)
├─ Historical Bias           └─ Representation Bias        └─ Anchoring Bias
│  (Data reflects past       │  (Model learns stereotypes     (First impression
│   discrimination)          │   from word/image              dominates)
└─ Label Bias                │   embeddings)
   (Annotator prejudice)     
```

#### Fairness Metrics

| Metric | Definition | Formula |
|--------|-----------|---------|
| **Demographic Parity** | Positive rate should be equal across groups | P(Ŷ=1\|A=0) = P(Ŷ=1\|A=1) |
| **Equalized Odds** | True positive and false positive rates should be equal across groups | P(Ŷ=1\|Y=1,A=0) = P(Ŷ=1\|Y=1,A=1) |
| **Predictive Parity** | Precision should be equal across groups | P(Y=1\|Ŷ=1,A=0) = P(Y=1\|Ŷ=1,A=1) |
| **Individual Fairness** | Similar individuals should receive similar predictions | d(Ŷᵢ, Ŷⱼ) ≤ ε if d(xᵢ, xⱼ) ≤ δ |

> **⚠️ Impossibility Theorem (Chouldechova, 2017)**: It is mathematically impossible to simultaneously satisfy all fairness metrics when base rates differ between groups. This means **fairness is a design choice**, not a technical problem alone.

```python
"""
Bias Audit: Detecting Gender Bias in Income Prediction
Dataset: UCI Adult Census Income (real data, 48,842 records)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Adult Census dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
df = df.dropna()

print(f"Dataset size: {len(df)} records")
print(f"\nIncome distribution:")
print(df['income'].value_counts(normalize=True))
print(f"\nGender distribution:")
print(df['sex'].value_counts())

# Prepare features
le = LabelEncoder()
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# ─── BIAS AUDIT ──────────────────────────────────────
# Check prediction fairness across gender
test_df = X_test.copy()
test_df['true_income'] = y_test.values
test_df['pred_income'] = y_pred
test_df['sex_label'] = df.loc[X_test.index, 'sex'].values

print("\n" + "=" * 60)
print("BIAS AUDIT: Gender Fairness Analysis")
print("=" * 60)

for gender in ['Male', 'Female']:
    mask = test_df['sex_label'] == gender
    group = test_df[mask]
    
    # Demographic Parity: rate of predicting high income
    positive_rate = (group['pred_income'] == 1).mean()
    
    # Accuracy for this group
    group_accuracy = accuracy_score(group['true_income'], group['pred_income'])
    
    # True Positive Rate (Equalized Odds component)
    true_positives = group[(group['true_income'] == 1) & (group['pred_income'] == 1)]
    actual_positives = group[group['true_income'] == 1]
    tpr = len(true_positives) / len(actual_positives) if len(actual_positives) > 0 else 0
    
    # False Positive Rate
    false_positives = group[(group['true_income'] == 0) & (group['pred_income'] == 1)]
    actual_negatives = group[group['true_income'] == 0]
    fpr = len(false_positives) / len(actual_negatives) if len(actual_negatives) > 0 else 0
    
    print(f"\n{gender}:")
    print(f"  Group size:          {len(group)}")
    print(f"  Accuracy:            {group_accuracy:.2%}")
    print(f"  Positive pred rate:  {positive_rate:.2%} (Demographic Parity)")
    print(f"  True Positive Rate:  {tpr:.2%} (Equalized Odds)")
    print(f"  False Positive Rate: {fpr:.2%} (Equalized Odds)")

print(f"\n{'=' * 60}")
print("⚠️  If positive prediction rates differ significantly between")
print("   genders, the model may have a fairness issue.")
print("   This doesn't mean the MODEL is biased — it may reflect")
print("   bias in the TRAINING DATA (historical income inequality).")
print(f"{'=' * 60}")
```

### 🔴 Advanced Level

#### Ethical Frameworks for AI

| Framework | Description | Key Principles |
|-----------|-------------|----------------|
| **EU AI Act (2024)** | World's first comprehensive AI regulation | Risk-based: Unacceptable → High → Limited → Minimal |
| **IEEE Ethically Aligned Design** | Engineering standards for ethical AI | Transparency, accountability, awareness of misuse |
| **OECD AI Principles** | International policy guidelines | Inclusive growth, human-centred values, transparency |
| **Google AI Principles** | Corporate AI ethics guidelines | Be socially beneficial, avoid creating unfair bias, be accountable |

#### Bias Mitigation Strategies

| Stage | Strategy | Description |
|-------|----------|-------------|
| **Pre-processing** | Reweighting | Adjust sample weights to balance protected groups |
| **Pre-processing** | Data augmentation | Generate synthetic data for underrepresented groups |
| **In-processing** | Adversarial debiasing | Add adversary network that penalizes biased predictions |
| **In-processing** | Fairness constraints | Add fairness metric as optimization constraint |
| **Post-processing** | Threshold adjustment | Different decision thresholds per group |
| **Post-processing** | Reject option classification | Defer uncertain decisions to humans |

**Reference**: Mehrabi, N., et al. (2021). "A Survey on Bias and Fairness in Machine Learning." *ACM Computing Surveys*, 54(6), 1-35.

---

## Part 7: Responsible AI Practices

### 🟢 Beginner Level

**Responsible AI** means developing and deploying AI systems that are:

| Principle | What It Means | Example |
|-----------|---------------|---------|
| **Transparent** | People understand how the AI makes decisions | Explaining why a loan was denied |
| **Fair** | AI doesn't discriminate against protected groups | Equal opportunity across genders/races |
| **Accountable** | Someone is responsible for the AI's decisions | Clear ownership and audit trails |
| **Private** | Personal data is protected | GDPR compliance, anonymization |
| **Safe** | AI doesn't cause harm | Medical AI with human oversight |

### 🟡 Intermediate Level — Implementing Responsible AI

```python
"""
Responsible AI Checklist Implementation
This shows how to add basic responsible AI checks to any ML project.
"""
import pandas as pd
import numpy as np

def responsible_ai_checklist(X, y, model, protected_attributes, X_test, y_test):
    """
    Run a basic Responsible AI audit on a trained model.
    
    Parameters
    ----------
    X : DataFrame - Training features
    y : Series - Training labels
    model : trained sklearn model
    protected_attributes : dict - {attr_name: [group_values]}
    X_test : DataFrame - Test features
    y_test : Series - Test labels
    
    Returns
    -------
    dict : Audit results
    """
    results = {"checks": []}
    y_pred = model.predict(X_test)
    
    # Check 1: Data representation
    print("═══ CHECK 1: Data Representation ═══")
    for attr, groups in protected_attributes.items():
        if attr in X.columns:
            for group in groups:
                pct = (X[attr] == group).mean()
                print(f"  {attr}={group}: {pct:.1%} of training data")
                if pct < 0.1:
                    print(f"    ⚠️  WARNING: Underrepresented group (<10%)")
    
    # Check 2: Performance parity
    print("\n═══ CHECK 2: Performance Across Groups ═══")
    for attr, groups in protected_attributes.items():
        if attr in X_test.columns:
            for group in groups:
                mask = X_test[attr] == group
                if mask.sum() > 0:
                    group_acc = (y_test[mask] == y_pred[mask]).mean()
                    print(f"  {attr}={group}: Accuracy = {group_acc:.2%}")
    
    # Check 3: Feature importance (transparency)
    print("\n═══ CHECK 3: Feature Importance (Transparency) ═══")
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print("  Top 5 features driving decisions:")
        for feat, imp in importances.head().items():
            flag = " ⚠️ PROTECTED" if feat in protected_attributes else ""
            print(f"    {feat}: {imp:.4f}{flag}")
    
    # Check 4: Prediction confidence
    print("\n═══ CHECK 4: Prediction Confidence ═══")
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        low_confidence = (proba.max(axis=1) < 0.6).mean()
        print(f"  Low confidence predictions (<60%): {low_confidence:.1%}")
        if low_confidence > 0.2:
            print("  ⚠️  WARNING: Too many uncertain predictions. Consider human review.")
    
    print("\n═══ AUDIT COMPLETE ═══")
    return results

# This function should be called after training any model in this course!
```

---

## 💻 Hands-On Lab

### Lab 1: Your First ML Pipeline (30 minutes)

**Objective**: Build a complete ML pipeline using real medical data.

**Dataset**: Scikit-learn's diabetes dataset

**Instructions**: See `code/01_ml_fundamentals.py` for the complete pipeline. Run it step by step:

```bash
cd Session_01_Introduction_to_ML/code
python 01_ml_fundamentals.py
```

Or open the Jupyter notebook:
```bash
cd Session_01_Introduction_to_ML/notebooks
jupyter notebook Session_01_Complete.ipynb
```

### Lab 2: Bias Audit (30 minutes)

**Objective**: Audit a real ML model for gender and race bias.

**Dataset**: UCI Adult Census Income dataset

**Instructions**: See `code/03_bias_audit.py`

```bash
python 03_bias_audit.py
```

---

## 📊 Portfolio Task

### "Responsible AI Audit Report"

Create a professional audit report that includes:

1. **Dataset Description**: What data was used, its source, and limitations
2. **Model Performance**: Overall accuracy, precision, recall, F1
3. **Fairness Analysis**: Demographic parity and equalized odds across protected groups
4. **Findings**: What biases were detected? How significant are they?
5. **Recommendations**: How could the biases be mitigated?
6. **Conclusion**: Is this model safe to deploy?

**Template**: See `portfolio/portfolio_component.md`

> 💡 **Tip**: This report demonstrates to employers that you understand responsible AI — a highly valued skill in the industry.

---

## ✍️ Exercises

### 🟢 Beginner Exercises

See `exercises/exercises.md` for detailed exercises with starter code.

1. **Exercise 1.1**: Load the Iris dataset, split it 80/20, train a Decision Tree, and report accuracy
2. **Exercise 1.2**: Modify the ML pipeline to use different random_state values — how does accuracy change?
3. **Exercise 1.3**: List 5 real-world examples of ML applications you use daily

### 🟡 Intermediate Exercises

4. **Exercise 1.4**: Implement the full ML pipeline with `Pipeline` and `GridSearchCV`
5. **Exercise 1.5**: Compare 5 different algorithms on the California Housing dataset — which is best and why?
6. **Exercise 1.6**: Calculate demographic parity for the Adult Census model across race groups

### 🔴 Advanced Exercises

7. **Exercise 1.7**: Implement a bias mitigation strategy (reweighting or threshold adjustment) and measure improvement
8. **Exercise 1.8**: Write a model card for your diabetes prediction model following Google's model card template
9. **Exercise 1.9**: Research the EU AI Act — how would your model be classified? What requirements would apply?

---

## 📚 Further Reading

### Books
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly. **Chapter 1: The Machine Learning Landscape**
- Raschka, S. & Mirjalili, V. (2022). *Python Machine Learning* (3rd ed.). Packt. **Chapter 1-2**

### Papers
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill. (Original ML definition)
- Mehrabi, N. et al. (2021). "A Survey on Bias and Fairness in Machine Learning." *ACM Computing Surveys*
- Chouldechova, A. (2017). "Fair Prediction with Disparate Impact." *Big Data*, 5(2), 153-163.

### Online Resources
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Google Model Cards](https://modelcards.withgoogle.com/)

---

## ⬅️ Previous | [🏠 Home](../README.md) | [Session 02: Advanced Clustering ➡️](../Session_02_Advanced_Clustering/)
