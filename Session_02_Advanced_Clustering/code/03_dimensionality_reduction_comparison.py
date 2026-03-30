"""
============================================================
03_dimensionality_reduction_comparison.py
Applied Machine Learning Using Python — Session 02
============================================================
Topic: PCA vs t-SNE vs UMAP Comparison
Level: 🟡 Intermediate → 🔴 Advanced

Comprehensive comparison of dimensionality reduction techniques:
1. PCA — linear, fast, preserves global structure
2. t-SNE — non-linear, slow, preserves local neighborhoods
3. UMAP — non-linear, fast, preserves local + some global

Dataset: Scikit-learn Digits (1,797 images, 64 dimensions)

Usage:
    python 03_dimensionality_reduction_comparison.py
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time

plt.style.use("seaborn-v0_8-whitegrid")


def print_header(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def main():
    # ══════════════════════════════════════════════════════════
    # STEP 1: Load & Explore High-Dimensional Data
    # ══════════════════════════════════════════════════════════
    print_header("STEP 1: Loading High-Dimensional Data")
    
    digits = load_digits()
    X = digits.data  # 1,797 × 64
    y = digits.target  # 0-9
    
    print(f"Dataset: Handwritten Digits (8×8 pixel images)")
    print(f"  Samples:    {X.shape[0]}")
    print(f"  Dimensions: {X.shape[1]}")
    print(f"  Classes:    {len(np.unique(y))} (digits 0-9)")
    
    # Show sample images
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    fig.suptitle("Sample Handwritten Digits (8×8 pixels = 64 features)", fontsize=14)
    for i in range(20):
        row, col = divmod(i, 10)
        axes[row, col].imshow(digits.images[i], cmap="gray")
        axes[row, col].set_title(str(digits.target[i]), fontsize=10)
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig("../notebooks/sample_digits.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: PCA Analysis
    # ══════════════════════════════════════════════════════════
    print_header("STEP 2: PCA (Principal Component Analysis)")
    
    # Full PCA to analyze variance
    pca_full = PCA().fit(X_scaled)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Key thresholds
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_components = np.argmax(cumulative_var >= threshold) + 1
        print(f"  Components for {threshold:.0%} variance: {n_components}")
    
    # PCA to 2D
    t_start = time.time()
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    pca_time = time.time() - t_start
    print(f"\n  PCA (2D) completed in {pca_time:.3f} seconds")
    print(f"  Variance retained: {pca_2d.explained_variance_ratio_.sum():.1%}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: t-SNE Analysis
    # ══════════════════════════════════════════════════════════
    print_header("STEP 3: t-SNE (t-distributed Stochastic Neighbor Embedding)")
    
    t_start = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000,
                learning_rate="auto", init="pca")
    X_tsne_2d = tsne.fit_transform(X_scaled)
    tsne_time = time.time() - t_start
    print(f"  t-SNE completed in {tsne_time:.3f} seconds")
    print(f"  Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    # Compare different perplexity values
    print(f"\n  Effect of perplexity parameter:")
    perplexities = [5, 15, 30, 50]
    tsne_results = {}
    for perp in perplexities:
        tsne_p = TSNE(n_components=2, random_state=42, perplexity=perp,
                       n_iter=1000, learning_rate="auto", init="pca")
        tsne_results[perp] = tsne_p.fit_transform(X_scaled)
        print(f"    Perplexity={perp:3d}: KL divergence = {tsne_p.kl_divergence_:.4f}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: UMAP Analysis (if available)
    # ══════════════════════════════════════════════════════════
    print_header("STEP 4: UMAP (Uniform Manifold Approximation and Projection)")
    
    has_umap = False
    try:
        import umap
        t_start = time.time()
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap_2d = reducer.fit_transform(X_scaled)
        umap_time = time.time() - t_start
        has_umap = True
        print(f"  UMAP completed in {umap_time:.3f} seconds")
    except ImportError:
        print("  ⚠️  UMAP not installed. Install with: pip install umap-learn")
        print("  Continuing with PCA and t-SNE comparison only.")
        X_umap_2d = None
        umap_time = None
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: Comparison Visualization
    # ══════════════════════════════════════════════════════════
    print_header("STEP 5: Visual Comparison")
    
    n_methods = 3 if has_umap else 2
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 6))
    fig.suptitle("Dimensionality Reduction: 64D → 2D", fontsize=18, fontweight="bold")
    
    # PCA
    scatter = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap="tab10",
                               alpha=0.5, s=10)
    axes[0].set_title(f"PCA\nTime: {pca_time:.3f}s | "
                       f"Variance: {pca_2d.explained_variance_ratio_.sum():.0%}")
    
    # t-SNE
    axes[1].scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, cmap="tab10",
                     alpha=0.5, s=10)
    axes[1].set_title(f"t-SNE (perplexity=30)\nTime: {tsne_time:.3f}s")
    
    # UMAP
    if has_umap:
        axes[2].scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c=y, cmap="tab10",
                         alpha=0.5, s=10)
        axes[2].set_title(f"UMAP\nTime: {umap_time:.3f}s")
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[-1], label="Digit")
    cbar.set_ticks(range(10))
    
    plt.tight_layout()
    plt.savefig("../notebooks/dim_reduction_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: t-SNE Perplexity Comparison
    # ══════════════════════════════════════════════════════════
    print_header("STEP 6: t-SNE Perplexity Effect")
    
    fig, axes = plt.subplots(1, len(perplexities), figsize=(5 * len(perplexities), 5))
    fig.suptitle("Effect of Perplexity on t-SNE", fontsize=16, fontweight="bold")
    
    for ax, perp in zip(axes, perplexities):
        ax.scatter(tsne_results[perp][:, 0], tsne_results[perp][:, 1],
                    c=y, cmap="tab10", alpha=0.5, s=10)
        ax.set_title(f"Perplexity = {perp}")
    
    plt.tight_layout()
    plt.savefig("../notebooks/tsne_perplexity.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # ══════════════════════════════════════════════════════════
    # STEP 7: PCA for Preprocessing (Speed Improvement)
    # ══════════════════════════════════════════════════════════
    print_header("STEP 7: PCA as Preprocessing for Clustering")
    
    # Cluster in original 64D space
    t_start = time.time()
    kmeans_64d = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels_64d = kmeans_64d.fit_predict(X_scaled)
    time_64d = time.time() - t_start
    sil_64d = silhouette_score(X_scaled, labels_64d)
    
    # Cluster in PCA-reduced space (95% variance)
    pca_95 = PCA(n_components=0.95)
    X_pca_95 = pca_95.fit_transform(X_scaled)
    
    t_start = time.time()
    kmeans_pca = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels_pca = kmeans_pca.fit_predict(X_pca_95)
    time_pca = time.time() - t_start
    sil_pca = silhouette_score(X_pca_95, labels_pca)
    
    print(f"\n  {'Metric':<25s} {'64D (Original)':>15s} {'PCA ({0}D)'.format(pca_95.n_components_):>15s}")
    print(f"  {'─' * 55}")
    print(f"  {'Dimensions':<25s} {64:>15d} {pca_95.n_components_:>15d}")
    print(f"  {'Clustering Time':<25s} {time_64d:>15.4f}s {time_pca:>15.4f}s")
    print(f"  {'Silhouette Score':<25s} {sil_64d:>15.4f} {sil_pca:>15.4f}")
    print(f"  {'Speedup':<25s} {'':>15s} {time_64d/max(time_pca, 0.001):>14.1f}×")
    
    print(f"\n  💡 PCA reduced dimensions from 64 → {pca_95.n_components_} while keeping")
    print(f"     95% of variance, making clustering faster with similar quality!")
    
    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print_header("SUMMARY: When to Use What")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  Method │ Use When                                      │
  ├─────────┼───────────────────────────────────────────────┤
  │  PCA    │ • Preprocessing before ML (reduce features)   │
  │         │ • Need to preserve variance                   │
  │         │ • Fast computation required                   │
  │         │ • Data has linear relationships               │
  ├─────────┼───────────────────────────────────────────────┤
  │  t-SNE  │ • 2D/3D visualization ONLY                    │
  │         │ • Exploring cluster structure                  │
  │         │ • Small-medium datasets (< 10K)               │
  │         │ • ⚠️ DON'T use for preprocessing!             │
  ├─────────┼───────────────────────────────────────────────┤
  │  UMAP   │ • Visualization (fast alternative to t-SNE)   │
  │         │ • CAN be used for preprocessing               │
  │         │ • Large datasets                              │
  │         │ • Preserves more global structure than t-SNE   │
  └─────────┴───────────────────────────────────────────────┘
    """)
    
    print("✅ Dimensionality Reduction Comparison Complete!")


if __name__ == "__main__":
    main()
