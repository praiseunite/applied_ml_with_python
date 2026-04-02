"""
============================================================
02_dbscan_anomaly_detection.py
Applied Machine Learning Using Python — Session 02
============================================================
Topic: DBSCAN for Anomaly Detection & Non-Spherical Clustering
Level: 🟡 Intermediate

Demonstrates:
1. DBSCAN on non-spherical data (where K-Means fails)
2. Parameter tuning with k-distance graph
3. Anomaly/outlier detection using DBSCAN
4. Comparison: K-Means vs DBSCAN vs Hierarchical

Usage:
    python 02_dbscan_anomaly_detection.py
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

plt.style.use("seaborn-v0_8-whitegrid")


def print_header(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def main():
    # ══════════════════════════════════════════════════════════
    # PART 1: Why K-Means Fails on Non-Spherical Data
    # ══════════════════════════════════════════════════════════
    print_header("PART 1: K-Means Failures on Non-Spherical Data")
    
    # Generate different data shapes
    datasets = {
        "Moons": make_moons(n_samples=500, noise=0.05, random_state=42),
        "Circles": make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42),
        "Blobs": make_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42),
        "Anisotropic": None,  # Will generate below
    }
    
    # Generate anisotropic (stretched) data
    X_aniso, y_aniso = make_blobs(n_samples=500, centers=3, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    datasets["Anisotropic"] = (X_aniso, y_aniso)
    
    algorithms = {
        "K-Means": lambda X, k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X),
        "DBSCAN": lambda X, _: DBSCAN(eps=0.3, min_samples=5).fit_predict(X),
        "Hierarchical": lambda X, k: AgglomerativeClustering(n_clusters=k).fit_predict(X),
    }
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle("Algorithm Comparison on Different Data Shapes", fontsize=18, fontweight="bold")
    
    for row, (data_name, (X, y)) in enumerate(datasets.items()):
        X_scaled = StandardScaler().fit_transform(X)
        n_true_clusters = len(np.unique(y))
        
        # Original data
        axes[row, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="tab10",
                              alpha=0.6, s=15)
        axes[row, 0].set_title(f"{data_name}\n(Ground Truth)" if row == 0 
                                else f"{data_name}")
        axes[row, 0].set_ylabel(data_name, fontsize=12, fontweight="bold")
        
        # Apply each algorithm
        for col, (algo_name, algo_func) in enumerate(algorithms.items(), 1):
            try:
                labels = algo_func(X_scaled, n_true_clusters)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                
                # Color noise points black
                colors = labels.copy().astype(float)
                colors[labels == -1] = -1
                
                axes[row, col].scatter(X_scaled[:, 0], X_scaled[:, 1],
                                        c=colors, cmap="tab10", alpha=0.6, s=15)
                
                # Calculate silhouette if valid
                valid_mask = labels >= 0
                if len(np.unique(labels[valid_mask])) > 1:
                    sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
                    info = f"Sil: {sil:.2f}"
                else:
                    info = "N/A"
                
                subtitle = f"{n_clusters} clusters"
                if n_noise > 0:
                    subtitle += f", {n_noise} noise"
                
                axes[row, col].set_title(f"{algo_name}\n{subtitle} ({info})" if row == 0
                                          else f"{subtitle} ({info})")
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"Error:\n{str(e)[:30]}",
                                     ha="center", va="center")
        
        if row == 0:
            axes[0, 0].set_title(f"Ground Truth", fontsize=11)
            for col, name in enumerate(algorithms.keys(), 1):
                axes[0, col].set_title(f"{name}\n{axes[0, col].get_title()}", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("../notebooks/algorithm_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # ══════════════════════════════════════════════════════════
    # PART 2: DBSCAN Parameter Tuning
    # ══════════════════════════════════════════════════════════
    print_header("PART 2: DBSCAN Parameter Tuning")
    
    X_moons, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
    X_moons_scaled = StandardScaler().fit_transform(X_moons)
    
    # k-distance graph for eps selection
    k_values = [3, 5, 7, 10]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DBSCAN Parameter Tuning", fontsize=16, fontweight="bold")
    
    for k in k_values:
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_moons_scaled)
        distances, _ = nn.kneighbors(X_moons_scaled)
        distances = np.sort(distances[:, k - 1])
        axes[0].plot(distances, label=f"k={k}", linewidth=1.5)
    
    axes[0].set_title("K-Distance Graph")
    axes[0].set_xlabel("Points (sorted)")
    axes[0].set_ylabel("Distance to k-th neighbor")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Try different eps values
    eps_values = np.arange(0.1, 1.0, 0.05)
    results = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_moons_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        valid = labels >= 0
        sil = silhouette_score(X_moons_scaled[valid], labels[valid]) if np.sum(valid) > 1 and len(np.unique(labels[valid])) > 1 else 0
        
        results.append({"eps": eps, "clusters": n_clusters, "noise": n_noise, "silhouette": sil})
    
    results_df = pd.DataFrame(results)
    axes[1].plot(results_df["eps"], results_df["clusters"], "b-o", label="Clusters", markersize=4)
    ax2 = axes[1].twinx()
    ax2.plot(results_df["eps"], results_df["noise"], "r-s", label="Noise points", markersize=4)
    axes[1].set_xlabel("eps")
    axes[1].set_ylabel("Number of Clusters", color="blue")
    ax2.set_ylabel("Noise Points", color="red")
    axes[1].set_title("Effect of eps on DBSCAN")
    axes[1].legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("../notebooks/dbscan_tuning.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # ══════════════════════════════════════════════════════════
    # PART 3: Anomaly Detection with DBSCAN
    # ══════════════════════════════════════════════════════════
    print_header("PART 3: Anomaly Detection with DBSCAN")
    
    # Generate normal data with some anomalies
    np.random.seed(42)
    X_normal = np.vstack([
        np.random.normal(0, 1, (200, 2)),
        np.random.normal(4, 0.8, (150, 2)),
    ])
    
    # Add anomalies
    X_anomalies = np.random.uniform(-3, 7, (20, 2))
    X_all = np.vstack([X_normal, X_anomalies])
    true_labels = np.array([0] * 200 + [1] * 150 + [-1] * 20)
    
    X_all_scaled = StandardScaler().fit_transform(X_all)
    
    # DBSCAN for anomaly detection
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_all_scaled)
    
    n_anomalies_detected = (labels == -1).sum()
    true_anomalies = 20
    
    print(f"  Total points:       {len(X_all)}")
    print(f"  True anomalies:     {true_anomalies}")
    print(f"  Detected anomalies: {n_anomalies_detected}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Anomaly Detection with DBSCAN", fontsize=16, fontweight="bold")
    
    ax1.scatter(X_all[:, 0], X_all[:, 1], c=true_labels, cmap="RdYlGn",
                alpha=0.6, s=30)
    ax1.set_title("Ground Truth (Red = Anomalies)")
    
    normal = labels >= 0
    anomaly = labels == -1
    ax2.scatter(X_all[normal, 0], X_all[normal, 1], c=labels[normal],
                cmap="viridis", alpha=0.6, s=30, label="Normal")
    ax2.scatter(X_all[anomaly, 0], X_all[anomaly, 1], c="red", marker="x",
                s=50, linewidths=2, label="Anomaly (DBSCAN)")
    ax2.set_title(f"DBSCAN Detection ({n_anomalies_detected} anomalies found)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("../notebooks/anomaly_detection.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print("\n✅ DBSCAN Analysis Complete!")
    print("   Key takeaways:")
    print("   1. DBSCAN handles non-spherical clusters that K-Means can't")
    print("   2. The k-distance graph helps find the right eps parameter")
    print("   3. Points labeled -1 are anomalies/outliers")
    print("   4. DBSCAN doesn't require specifying K in advance")


if __name__ == "__main__":
    main()
