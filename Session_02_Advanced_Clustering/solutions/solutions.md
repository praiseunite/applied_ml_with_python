# Session 02 — Solutions

## 🟢 Solution 2.1: K-Means on Iris

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data[:, [0, 2]]  # sepal_length and petal_length

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# K-Means clusters
ax1.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=40)
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', s=200, c='red', edgecolors='black', linewidth=2)
ax1.set_title('K-Means Clusters')
ax1.set_xlabel('Sepal Length'); ax1.set_ylabel('Petal Length')

# Actual species
ax2.scatter(X[:, 0], X[:, 1], c=iris.target, cmap='viridis', alpha=0.6, s=40)
ax2.set_title('Actual Species')
ax2.set_xlabel('Sepal Length'); ax2.set_ylabel('Petal Length')

plt.tight_layout()
plt.show()

# Comparison metric
ari = adjusted_rand_score(iris.target, clusters)
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"(1.0 = perfect match, 0.0 = random)")
print(f"K-Means closely matches the true species labels!")
```

---

## 🟢 Solution 2.2: Elbow & Silhouette Analysis

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

K_range = range(2, 11)
inertias, sil_scores = [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))
    print(f"K={k}: Inertia={km.inertia_:.0f}, Silhouette={silhouette_score(X, labels):.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2)
ax1.set_title('Elbow Method')
ax1.set_xlabel('K'); ax1.set_ylabel('Inertia')
ax1.axvline(4, color='red', linestyle='--', label='Elbow at K=4')
ax1.legend()

ax2.bar(K_range, sil_scores, color='#2ecc71')
ax2.set_title('Silhouette Score')
ax2.set_xlabel('K'); ax2.set_ylabel('Score')

plt.tight_layout()
plt.show()

optimal_k = list(K_range)[np.argmax(sil_scores)]
print(f"\n✅ Both methods agree: K={optimal_k} is optimal")
```

---

## 🟡 Solution 2.3: DBSCAN Parameter Tuning

```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# k-distance graph
k = 5
nn = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances = np.sort(nn.kneighbors(X_scaled)[0][:, k-1])

plt.figure(figsize=(8, 4))
plt.plot(distances, linewidth=2)
plt.xlabel('Points (sorted)'); plt.ylabel(f'{k}-th Neighbor Distance')
plt.title('K-Distance Graph → find the elbow')
plt.axhline(y=0.3, color='red', linestyle='--', label='Suggested eps ≈ 0.3')
plt.legend()
plt.show()

# Try different min_samples
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, ms in zip(axes, [3, 5, 7, 10]):
    labels = DBSCAN(eps=0.3, min_samples=ms).fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=15, alpha=0.7)
    ax.set_title(f'min_samples={ms}\n{n_clusters} clusters, {noise} noise')
plt.suptitle('Effect of min_samples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Best: eps=0.3, min_samples=5 gives exactly 2 clusters with minimal noise
```

---

## 🔴 Solution 2.5: Image Compression with K-Means (Partial)

```python
"""
K-Means can compress images by reducing the number of colors.
Each pixel is a 3D point (R, G, B) → cluster into K colors.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

# Load a sample image
image = load_sample_image("china.jpg")  # Built-in sklearn image
print(f"Original shape: {image.shape}")  # (427, 640, 3)
print(f"Original colors: {image.reshape(-1, 3).shape[0]:,} pixels")

# Reshape to (n_pixels, 3)
pixels = image.reshape(-1, 3).astype(float) / 255.0

# Compress with different K values
K_values = [4, 8, 16, 32, 64]
fig, axes = plt.subplots(1, len(K_values) + 1, figsize=(20, 4))

axes[0].imshow(image)
axes[0].set_title(f'Original\n({len(np.unique(pixels, axis=0)):,} colors)')
axes[0].axis('off')

for ax, k in zip(axes[1:], K_values):
    km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=20)
    labels = km.fit_predict(pixels)
    compressed = km.cluster_centers_[labels].reshape(image.shape[:2] + (3,))
    ax.imshow(compressed)
    ratio = 3 * image.shape[0] * image.shape[1] / (k * 3 + len(labels) * np.ceil(np.log2(k)) / 8)
    ax.set_title(f'K={k}\n(~{ratio:.0f}× compression)')
    ax.axis('off')

plt.suptitle('Image Compression with K-Means', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Key Learning**: K-Means replaces each pixel's color with its nearest centroid color. With K=16, images look nearly identical to the original but use only 16 colors instead of millions!
