# Session 02 — Exercises

## 🟢 Beginner Exercises

### Exercise 2.1: K-Means on Iris

**Objective**: Apply K-Means clustering to a familiar dataset.

1. Load the Iris dataset (use only `sepal_length` and `petal_length` for 2D visualization)
2. Apply K-Means with K=3
3. Visualize the clusters with different colors
4. Compare your clusters with the actual species labels — how well did K-Means do?

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, [0, 2]]  # sepal_length and petal_length only

# TODO: Apply KMeans with 3 clusters
# TODO: Plot clusters
# TODO: Compare with actual labels
```

### Exercise 2.2: Elbow & Silhouette Analysis

1. Generate blob data with `make_blobs(n_samples=500, centers=4, random_state=42)`
2. Run K-Means for K=2 to K=10
3. Plot both the Elbow curve and Silhouette scores
4. What K do they suggest? Do they agree?

---

## 🟡 Intermediate Exercises

### Exercise 2.3: DBSCAN Parameter Tuning

1. Generate moon-shaped data: `make_moons(n_samples=500, noise=0.1)`
2. Use the k-distance graph to find the optimal `eps`
3. Try `min_samples` values of 3, 5, 7, 10 — how does it affect results?
4. Find the parameter combination that produces exactly 2 clusters with minimal noise

### Exercise 2.4: Algorithm Showdown

1. Generate 4 different datasets: blobs, moons, circles, and anisotropic blobs
2. Apply K-Means, DBSCAN, Hierarchical, and GMM to each
3. Create a 4×4 grid of subplots showing all combinations
4. Rate each algorithm on each dataset (✅ good / ⚠️ okay / ❌ bad)

---

## 🔴 Advanced Exercises

### Exercise 2.5: Image Compression with K-Means

**Objective**: Use K-Means to compress a real image by reducing the number of colors.

1. Load any image using `matplotlib.image.imread()` (or use a sample image)
2. Reshape the image from (H, W, 3) to (H×W, 3) — each pixel is a 3D point (RGB)
3. Apply K-Means with K = 4, 8, 16, 32, 64 clusters
4. Replace each pixel with its cluster centroid color
5. Display the original and compressed images side by side
6. Calculate the compression ratio

### Exercise 2.6: Clustering Evaluation Pipeline

Build a reusable function that takes any dataset and automatically:
1. Scales the data
2. Tries K-Means (K=2..10), DBSCAN (multiple eps), and Hierarchical clustering
3. Evaluates each with silhouette score
4. Returns a ranked comparison table
5. Generates a visualization of the best result

---

## 📝 Submission

Submit your code as `.py` files with all visualizations saved as PNG files.
