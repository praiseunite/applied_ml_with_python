# Session 02: Unsupervised Advanced Clustering Algorithms

<p align="center">
  <img src="https://img.shields.io/badge/Duration-2%20Hours-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TL3-Session%203-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Level-All%20Levels-green?style=for-the-badge" />
</p>

> **Covers**: TL3 (Session 3) — 2 hours
>
> **Book Reference**: Applied Machine Learning Using Python — Session 2

---

## 🎯 Learning Objectives

| # | Objective | Level |
|---|-----------|-------|
| 1 | Define clustering, its tools, and application scenarios | 🟢 Beginner |
| 2 | Describe different clustering techniques and practical implementation | 🟡 Intermediate |
| 3 | List and explain different dimensionality reduction techniques | 🔴 Advanced |

---

## 📋 Prerequisites

- Session 01 completed
- Understanding of supervised vs unsupervised learning
- Basic numpy, pandas, and matplotlib skills

---

## Table of Contents

- [Part 1: What is Clustering?](#part-1-what-is-clustering)
- [Part 2: K-Means Clustering](#part-2-k-means-clustering)
- [Part 3: DBSCAN](#part-3-dbscan)
- [Part 4: Hierarchical Clustering](#part-4-hierarchical-clustering)
- [Part 5: Gaussian Mixture Models](#part-5-gaussian-mixture-models)
- [Part 6: Dimensionality Reduction](#part-6-dimensionality-reduction)
- [Part 7: Choosing the Right Algorithm](#part-7-choosing-the-right-algorithm)
- [Hands-On Lab](#-hands-on-lab)
- [Exercises](#-exercises)

---

## Part 1: What is Clustering?

### 🟢 Beginner Level

**Clustering** is an unsupervised learning technique that groups similar data points together **without any labels**. The algorithm discovers patterns on its own.

#### Real-World Analogy

Imagine you're sorting a pile of 1,000 photos with no labels:
- You'd naturally group them: landscapes, portraits, food, animals
- You didn't need labels — you found patterns based on visual similarity  
- That's clustering!

#### Why Clustering Matters

| Application | What's Being Clustered | Business Value |
|-------------|----------------------|----------------|
| **Customer Segmentation** | Customers by behavior | Targeted marketing, personalized experiences |
| **Document Organization** | Text documents by topic | Automatic categorization, search improvement |
| **Image Grouping** | Images by visual similarity | Photo organization, content moderation |
| **Anomaly Detection** | Data points by normality | Fraud detection, security monitoring |
| **Gene Expression** | Genes by expression patterns | Drug discovery, disease research |
| **Social Network Analysis** | Users by connections | Community detection, recommendations |

#### How Do We Measure "Similarity"?

Clustering relies on **distance metrics** — mathematical ways to measure how "close" two data points are:

| Metric | Formula | Best For |
|--------|---------|----------|
| **Euclidean** | √(Σ(xᵢ - yᵢ)²) | Continuous data, when all features have similar scales |
| **Manhattan** | Σ\|xᵢ - yᵢ\| | Grid-like data, when outliers are present |
| **Cosine** | 1 - (x·y)/(‖x‖·‖y‖) | Text data, high-dimensional sparse data |

```python
"""
Distance Metrics Comparison
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances

# Two customer profiles
customer_a = np.array([[25, 50000, 3]])   # Age, Income, Purchases
customer_b = np.array([[30, 52000, 5]])   # Age, Income, Purchases

print("Customer A:", customer_a[0])
print("Customer B:", customer_b[0])
print()
print(f"Euclidean Distance: {euclidean_distances(customer_a, customer_b)[0][0]:.2f}")
print(f"Manhattan Distance: {manhattan_distances(customer_a, customer_b)[0][0]:.2f}")
print(f"Cosine Distance:    {cosine_distances(customer_a, customer_b)[0][0]:.6f}")

# ⚠️ Notice: Euclidean distance is dominated by Income (50000 vs 52000)
# This is why FEATURE SCALING is critical for clustering!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = np.vstack([customer_a, customer_b])
data_scaled = scaler.fit_transform(data)
print(f"\nAfter Scaling:")
print(f"Euclidean Distance: {euclidean_distances(data_scaled[:1], data_scaled[1:])[0][0]:.4f}")
```

---

## Part 2: K-Means Clustering

### 🟢 Beginner Level

**K-Means** is the most popular clustering algorithm. It divides data into **K** groups, where each point belongs to the nearest cluster center.

#### How K-Means Works

```
Step 1: Choose K (number of clusters)
Step 2: Randomly place K cluster centers (centroids)
Step 3: REPEAT until convergence:
    a) Assign each point to its nearest centroid
    b) Move each centroid to the mean of its assigned points
```

```python
"""
K-Means Clustering — Customer Segmentation
Dataset: Mall Customers (simulated real retail data)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Simulate Mall Customer data (similar to real Kaggle dataset)
np.random.seed(42)
n_customers = 200

# Create realistic customer segments
segment_params = [
    {"income": (30, 8), "spending": (20, 8), "n": 50},   # Low income, low spending
    {"income": (30, 8), "spending": (70, 10), "n": 40},  # Low income, high spending
    {"income": (60, 10), "spending": (50, 10), "n": 60}, # Medium income, medium spend
    {"income": (85, 8), "spending": (20, 8), "n": 25},   # High income, low spending
    {"income": (85, 8), "spending": (80, 8), "n": 25},   # High income, high spending
]

incomes, spendings = [], []
for params in segment_params:
    incomes.extend(np.random.normal(params["income"][0], params["income"][1], params["n"]))
    spendings.extend(np.random.normal(params["spending"][0], params["spending"][1], params["n"]))

customers = pd.DataFrame({
    "Annual_Income_K": np.clip(incomes, 10, 130),
    "Spending_Score": np.clip(spendings, 1, 100),
})

print(f"Customer Dataset: {len(customers)} customers")
print(customers.describe().round(1))

# Scale features (CRITICAL for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

# ─── Finding Optimal K: Elbow Method ───
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# ─── Finding Optimal K: Silhouette Score ───
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f"  K={k}: Inertia={inertias[k-2]:.1f}, Silhouette={sil_score:.3f}")

# ─── Fit final model with optimal K ───
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n🏆 Optimal K = {optimal_k} (highest silhouette score)")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customers["Cluster"] = kmeans_final.fit_predict(X_scaled)

# ─── Visualize ───
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("K-Means Customer Segmentation", fontsize=16, fontweight="bold")

# Elbow Method
axes[0].plot(K_range, inertias, "bo-", linewidth=2)
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-Cluster SSE)")
axes[0].axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal K={optimal_k}")
axes[0].legend()

# Silhouette Scores
axes[1].bar(K_range, silhouette_scores, color="#2ecc71", edgecolor="white")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(x=optimal_k, color="red", linestyle="--")

# Final Clusters
scatter = axes[2].scatter(
    customers["Annual_Income_K"], customers["Spending_Score"],
    c=customers["Cluster"], cmap="viridis", alpha=0.6, s=50
)
centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
axes[2].scatter(centers[:, 0], centers[:, 1], marker="X", s=200, 
                c="red", edgecolors="black", linewidth=2, label="Centroids")
axes[2].set_title(f"Customer Segments (K={optimal_k})")
axes[2].set_xlabel("Annual Income ($K)")
axes[2].set_ylabel("Spending Score (1-100)")
axes[2].legend()

plt.tight_layout()
plt.savefig("kmeans_segmentation.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── Segment Analysis ───
print("\n📊 Segment Profiles:")
print("─" * 60)
for cluster in range(optimal_k):
    segment = customers[customers["Cluster"] == cluster]
    print(f"\nCluster {cluster} ({len(segment)} customers):")
    print(f"  Avg Income:   ${segment['Annual_Income_K'].mean():.0f}K")
    print(f"  Avg Spending:  {segment['Spending_Score'].mean():.0f}/100")
    
    # Business interpretation
    avg_income = segment['Annual_Income_K'].mean()
    avg_spending = segment['Spending_Score'].mean()
    if avg_income > 60 and avg_spending > 60:
        print(f"  → 💎 Premium Customers (high value, high spending)")
    elif avg_income > 60 and avg_spending < 40:
        print(f"  → 💰 Careful Spenders (high income, conservative)")
    elif avg_income < 40 and avg_spending > 60:
        print(f"  → 🛍️ Enthusiastic Shoppers (budget-conscious big spenders)")
    elif avg_income < 40 and avg_spending < 40:
        print(f"  → 📊 Standard Customers (budget-conscious, low spending)")
    else:
        print(f"  → 🎯 Average Customers (moderate income and spending)")
```

### 🟡 Intermediate Level — K-Means Limitations and Solutions

| Limitation | Why It Happens | Solution |
|-----------|---------------|----------|
| Must specify K in advance | Algorithm can't determine K automatically | Elbow method, silhouette analysis, gap statistic |
| Assumes spherical clusters | Uses distance to centroid (mean) | Use DBSCAN or GMM instead |
| Sensitive to initialization | Different starting centroids → different results | K-Means++ initialization (default in sklearn) |
| Sensitive to outliers | Outliers pull centroids away | Remove outliers first, or use K-Medoids |
| Only works with numeric data | Uses Euclidean distance | Encode categorical features, or use K-Prototypes |

---

## Part 3: DBSCAN

### 🟢 Beginner Level

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) finds clusters of any shape by looking at **density** of data points, not distance to a center.

#### Key Advantages over K-Means
- ✅ **No need to specify K** — discovers number of clusters automatically
- ✅ **Finds arbitrarily shaped clusters** — not just spherical
- ✅ **Identifies noise/outliers** — labels them as -1

#### Two Parameters
- **eps (ε)**: Maximum distance between two points to be considered neighbors
- **min_samples**: Minimum points required to form a dense region

```python
"""
DBSCAN vs K-Means — When shapes matter
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# Generate non-spherical data
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("K-Means vs DBSCAN: Cluster Shape Matters", fontsize=16, fontweight="bold")

for row, (X, name) in enumerate([(X_moons, "Moons"), (X_circles, "Circles")]):
    X_scaled = StandardScaler().fit_transform(X)
    
    # Original data
    axes[row, 0].scatter(X[:, 0], X[:, 1], c="gray", alpha=0.6, s=30)
    axes[row, 0].set_title(f"{name} — Original Data")
    
    # K-Means (fails on non-spherical!)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    axes[row, 1].scatter(X[:, 0], X[:, 1], c=km_labels, cmap="viridis", alpha=0.6, s=30)
    axes[row, 1].set_title(f"{name} — K-Means ❌")
    
    # DBSCAN (handles any shape!)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    axes[row, 2].scatter(X[:, 0], X[:, 1], c=db_labels, cmap="viridis", alpha=0.6, s=30)
    noise = (db_labels == -1).sum()
    axes[row, 2].set_title(f"{name} — DBSCAN ✅ ({n_clusters} clusters, {noise} noise)")

plt.tight_layout()
plt.savefig("kmeans_vs_dbscan.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ DBSCAN correctly identifies non-spherical clusters where K-Means fails!")
```

### 🟡 Intermediate Level — Tuning DBSCAN Parameters

```python
"""
Finding optimal DBSCAN parameters using the k-distance graph
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Using the moons dataset from above
X_scaled = StandardScaler().fit_transform(X_moons)

# k-distance graph to find optimal eps
k = 5  # should match min_samples
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1])

plt.figure(figsize=(10, 5))
plt.plot(distances, linewidth=2)
plt.xlabel("Points (sorted by distance)")
plt.ylabel(f"{k}-th Nearest Neighbor Distance")
plt.title("K-Distance Graph — Find the 'Elbow' for Optimal eps")
plt.grid(True, alpha=0.3)
plt.axhline(y=0.3, color='red', linestyle='--', label='eps = 0.3')
plt.legend()
plt.tight_layout()
plt.show()
print("💡 The 'elbow' of this curve suggests the optimal eps value")
```

---

## Part 4: Hierarchical Clustering

### 🟢 Beginner Level

**Hierarchical clustering** builds a **tree of clusters** (called a **dendrogram**) by either:
- **Agglomerative** (bottom-up): Start with each point as its own cluster, merge closest pairs
- **Divisive** (top-down): Start with one big cluster, recursively split

```python
"""
Hierarchical Clustering with Dendrograms
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Create dendrogram (using a subset for readability)
np.random.seed(42)
sample_idx = np.random.choice(len(X), 30, replace=False)
X_sample = X[sample_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Hierarchical Clustering — Iris Dataset", fontsize=16, fontweight="bold")

# Dendrogram
linkage_matrix = linkage(X_sample, method="ward")
dendrogram(linkage_matrix, ax=axes[0], truncate_mode="level", p=5,
           leaf_font_size=8, above_threshold_color="gray")
axes[0].set_title("Dendrogram (Ward Linkage)")
axes[0].set_xlabel("Sample Index")
axes[0].set_ylabel("Distance")
axes[0].axhline(y=8, color='red', linestyle='--', label='Cut at 3 clusters')
axes[0].legend()

# Cluster visualization
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(X)
scatter = axes[1].scatter(X[:, 0], X[:, 2], c=labels, cmap="viridis", alpha=0.6, s=40)
axes[1].set_xlabel(feature_names[0])
axes[1].set_ylabel(feature_names[2])
axes[1].set_title("Clusters (3 groups)")
plt.colorbar(scatter, ax=axes[1], label="Cluster")

plt.tight_layout()
plt.savefig("hierarchical_clustering.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 🟡 Intermediate Level — Linkage Methods

| Linkage | How It Measures Distance | Best For |
|---------|-------------------------|----------|
| **Ward** | Minimizes variance increase when merging | Compact, equal-sized clusters |
| **Complete** | Maximum distance between points in two clusters | Finding well-separated clusters |
| **Average** | Average distance between all point pairs | Balanced approach |
| **Single** | Minimum distance between points in two clusters | Elongated clusters (chain-like) |

---

## Part 5: Gaussian Mixture Models

### 🟡 Intermediate Level

**GMMs** assume data comes from a **mixture of Gaussian distributions**. Unlike K-Means (hard assignment), GMMs provide **soft assignments** — each point has a probability of belonging to each cluster.

```python
"""
Gaussian Mixture Models — Soft Clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate overlapping clusters
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=[1.5, 1.0, 0.5],
                         random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

# BIC for model selection (finding optimal number of components)
bics = []
n_range = range(1, 8)
for n in n_range:
    g = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
    g.fit(X)
    bics.append(g.bic(X))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Gaussian Mixture Models", fontsize=16, fontweight="bold")

# Cluster assignment
axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6, s=30)
axes[0].set_title("GMM Cluster Assignments")

# Probability of belonging to cluster 0
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=probabilities[:, 0], 
                           cmap="RdYlGn", alpha=0.6, s=30)
plt.colorbar(scatter, ax=axes[1], label="P(Cluster 0)")
axes[1].set_title("Soft Assignment — P(Cluster 0)")

# BIC for model selection
axes[2].plot(n_range, bics, "bo-", linewidth=2)
axes[2].set_title("BIC Score (Lower = Better)")
axes[2].set_xlabel("Number of Components")
axes[2].set_ylabel("BIC")
optimal = n_range[np.argmin(bics)]
axes[2].axvline(x=optimal, color="red", linestyle="--", label=f"Optimal: {optimal}")
axes[2].legend()

plt.tight_layout()
plt.savefig("gmm_clustering.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 🔴 Advanced Level — Covariance Types

| Type | Shape | Parameters | Use When |
|------|-------|------------|----------|
| `full` | Ellipsoidal (any orientation) | K × D × D | Enough data, clusters have different shapes |
| `tied` | All clusters share same covariance | D × D | Clusters have similar shapes |
| `diag` | Axis-aligned ellipsoids | K × D | Features are independent |
| `spherical` | Spherical (like K-Means) | K × 1 | All features equally important |

---

## Part 6: Dimensionality Reduction

### 🟢 Beginner Level

**Dimensionality reduction** compresses high-dimensional data into fewer dimensions while preserving important information.

**Why?**
- **Visualization**: Can't visualize 50-dimensional data — reduce to 2D or 3D
- **Speed**: Fewer features = faster training
- **Noise removal**: Low-variance dimensions are often noise
- **Curse of dimensionality**: Too many features → poor performance

### PCA (Principal Component Analysis)

PCA finds new axes (principal components) that capture the **maximum variance** in the data.

```python
"""
PCA — Reducing Dimensions While Preserving Information
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load handwritten digits (64-dimensional data!)
digits = load_digits()
X = digits.data  # 1,797 samples × 64 features (8×8 pixel images)
y = digits.target

print(f"Original dimensions: {X.shape[1]}")

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA to 2D for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
print(f"After PCA (2D): {X_2d.shape[1]} dimensions")
print(f"Variance explained: {pca_2d.explained_variance_ratio_.sum():.1%}")

# PCA with 95% variance retained
pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X_scaled)
print(f"Components for 95% variance: {pca_95.n_components_}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("PCA — Dimensionality Reduction", fontsize=16, fontweight="bold")

# 2D projection
scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", 
                           alpha=0.5, s=10)
axes[0].set_title(f"64D → 2D ({pca_2d.explained_variance_ratio_.sum():.0%} variance)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.colorbar(scatter, ax=axes[0], label="Digit")

# Explained variance
pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
axes[1].plot(cumulative_variance, linewidth=2)
axes[1].axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
axes[1].axvline(x=pca_95.n_components_, color="green", linestyle="--",
                label=f"{pca_95.n_components_} components")
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance Ratio")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Scree plot
axes[2].bar(range(1, 21), pca_full.explained_variance_ratio_[:20], color="#3498db")
axes[2].set_title("Scree Plot (First 20 Components)")
axes[2].set_xlabel("Principal Component")
axes[2].set_ylabel("Variance Explained")

plt.tight_layout()
plt.savefig("pca_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 🔴 Advanced Level — t-SNE vs UMAP

| Method | Type | Speed | Preserves | Best For |
|--------|------|-------|-----------|----------|
| **PCA** | Linear | Fast | Global structure, variance | Preprocessing, feature reduction |
| **t-SNE** | Non-linear | Slow | Local neighborhoods | Visualization (2D/3D only) |
| **UMAP** | Non-linear | Fast | Local + some global structure | Visualization + preprocessing |

```python
"""
t-SNE vs UMAP Comparison on Digits Dataset
"""
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Non-Linear Dimensionality Reduction", fontsize=16, fontweight="bold")

scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.5, s=10)
axes[0].set_title("t-SNE")
plt.colorbar(scatter1, ax=axes[0], label="Digit")

# UMAP (if installed)
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    scatter2 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10", alpha=0.5, s=10)
    axes[1].set_title("UMAP")
    plt.colorbar(scatter2, ax=axes[1], label="Digit")
except ImportError:
    axes[1].text(0.5, 0.5, "UMAP not installed\npip install umap-learn", 
                 ha="center", va="center", fontsize=14)
    axes[1].set_title("UMAP (not installed)")

plt.tight_layout()
plt.savefig("tsne_umap_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Part 7: Choosing the Right Algorithm

### Decision Guide

```
Do you know the number of clusters?
│
├── YES
│   ├── Are clusters spherical/compact?
│   │   ├── YES → K-Means
│   │   └── NO
│   │       ├── Do clusters overlap? → GMM
│   │       └── Non-spherical shapes → Spectral Clustering
│   │
│   └── Need soft assignments? → GMM
│
└── NO
    ├── Do you need to detect outliers?
    │   ├── YES → DBSCAN
    │   └── NO
    │       ├── Want to visualize cluster hierarchy?
    │       │   ├── YES → Agglomerative (Dendrogram)
    │       │   └── NO → DBSCAN or HDBSCAN
    │       └── Very large dataset? → Mini-Batch K-Means
```

### Algorithm Comparison Summary

| Algorithm | Finds K? | Shapes | Outliers | Speed | Scalability |
|-----------|----------|--------|----------|-------|-------------|
| K-Means | ❌ | Spherical | ❌ | ⚡ Fast | Large datasets |
| DBSCAN | ✅ | Any | ✅ | 🔶 Medium | Medium datasets |
| Hierarchical | ❌ | Any | ❌ | 🐌 Slow | Small datasets |
| GMM | ❌ | Ellipsoidal | ❌ | 🔶 Medium | Medium datasets |
| HDBSCAN | ✅ | Any | ✅ | 🔶 Medium | Medium-Large |

---

## 💻 Hands-On Lab

### Lab: Customer Segmentation (45 minutes)

Run the complete customer segmentation pipeline:

```bash
cd Session_02_Advanced_Clustering/code
python 01_kmeans_customer_segmentation.py
```

### Lab: Algorithm Comparison (30 minutes)

Compare K-Means, DBSCAN, and Hierarchical on different data shapes:

```bash
python 02_dbscan_anomaly_detection.py
```

### Lab: Dimensionality Reduction (30 minutes)

Apply PCA and t-SNE to the digits dataset:

```bash
python 03_dimensionality_reduction_comparison.py
```

---

## 📊 Portfolio Task

### Customer Segmentation Dashboard

Build an interactive customer segmentation analysis:

1. Load a customer dataset (Mall Customers or generate realistic data)
2. Apply K-Means, DBSCAN, and Hierarchical clustering
3. Use PCA/t-SNE for visualization
4. Create segment profiles with business interpretations
5. **Deploy as a Streamlit app on Hugging Face Spaces** (Session 8)

See `portfolio/portfolio_component.md` for detailed instructions.

---

## ✍️ Exercises

See [exercises/exercises.md](exercises/exercises.md) for all exercises.

### Quick Summary

| Level | Exercise | Topic |
|-------|----------|-------|
| 🟢 | 2.1 | K-Means on Iris dataset |
| 🟢 | 2.2 | Elbow method and silhouette analysis |
| 🟡 | 2.3 | DBSCAN parameter tuning |
| 🟡 | 2.4 | Compare all 4 algorithms on same data |
| 🔴 | 2.5 | Image compression with K-Means |
| 🔴 | 2.6 | Build a clustering evaluation pipeline |

---

## 📚 Further Reading

### Books
- Géron, A. (2022). *Hands-On ML*. **Chapter 9: Unsupervised Learning Techniques**
- Raschka, S. *Python ML*. **Chapter 10: Working with Unlabeled Data — Clustering Analysis**

### Papers
- MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations." *Proc. 5th Berkeley Symp.*
- Ester, M. et al. (1996). "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." *KDD*
- van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*

---

## [⬅️ Session 01](../Session_01_Introduction_to_ML/) | [🏠 Home](../README.md) | [Session 03 ➡️](../Session_03_MDP_and_Reinforcement_Learning/)
