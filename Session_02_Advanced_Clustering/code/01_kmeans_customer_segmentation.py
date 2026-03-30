"""
============================================================
01_kmeans_customer_segmentation.py
Applied Machine Learning Using Python — Session 02
============================================================
Topic: K-Means Customer Segmentation
Level: 🟢 Beginner → 🟡 Intermediate

Complete customer segmentation pipeline:
1. Load/generate customer data
2. Exploratory Data Analysis
3. Feature scaling
4. Find optimal K (Elbow + Silhouette)
5. Apply K-Means
6. Segment profiling and business interpretation
7. Visualization dashboard

Usage:
    python 01_kmeans_customer_segmentation.py

Requirements:
    pip install numpy pandas scikit-learn matplotlib seaborn
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

plt.style.use("seaborn-v0_8-whitegrid")


def print_header(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def generate_customer_data(n_customers: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic customer data simulating a retail environment.
    
    Creates 5 distinct customer segments with realistic feature distributions.
    This simulates data similar to the Kaggle Mall Customers dataset.
    """
    np.random.seed(seed)
    
    segments = [
        {"name": "Budget Shoppers", "n": 60,
         "age": (22, 5), "income": (25, 8), "spending": (15, 8),
         "visits": (2, 1), "avg_purchase": (15, 5)},
        {"name": "Young Enthusiasts", "n": 50,
         "age": (25, 4), "income": (30, 10), "spending": (75, 10),
         "visits": (8, 2), "avg_purchase": (45, 15)},
        {"name": "Average Joes", "n": 80,
         "age": (40, 10), "income": (55, 12), "spending": (50, 12),
         "visits": (4, 2), "avg_purchase": (30, 10)},
        {"name": "Premium Cautious", "n": 55,
         "age": (50, 8), "income": (90, 10), "spending": (25, 10),
         "visits": (3, 1), "avg_purchase": (50, 20)},
        {"name": "VIP High-Value", "n": 55,
         "age": (35, 8), "income": (95, 12), "spending": (85, 8),
         "visits": (10, 3), "avg_purchase": (80, 25)},
    ]
    
    data = []
    for seg in segments:
        n = seg["n"]
        customers = pd.DataFrame({
            "Age": np.clip(np.random.normal(*seg["age"], n), 18, 70).astype(int),
            "Annual_Income_K": np.clip(np.random.normal(*seg["income"], n), 15, 140).round(1),
            "Spending_Score": np.clip(np.random.normal(*seg["spending"], n), 1, 100).astype(int),
            "Monthly_Visits": np.clip(np.random.normal(*seg["visits"], n), 1, 20).astype(int),
            "Avg_Purchase_Value": np.clip(np.random.normal(*seg["avg_purchase"], n), 5, 150).round(0),
        })
        data.append(customers)
    
    return pd.concat(data, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    # ══════════════════════════════════════════════════════════
    # STEP 1: Generate/Load Customer Data
    # ══════════════════════════════════════════════════════════
    print_header("STEP 1: Customer Data")
    
    customers = generate_customer_data(300)
    print(f"Total customers: {len(customers)}")
    print(f"\nFeatures:")
    for col in customers.columns:
        print(f"  {col:25s} | Mean: {customers[col].mean():>8.1f} | "
              f"Std: {customers[col].std():>7.1f} | "
              f"Range: [{customers[col].min():.0f}, {customers[col].max():.0f}]")
    
    print(f"\nCorrelation Matrix:")
    print(customers.corr().round(2).to_string())
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: EDA Visualizations
    # ══════════════════════════════════════════════════════════
    print_header("STEP 2: Exploratory Data Analysis")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Customer Data — Exploratory Analysis", fontsize=16, fontweight="bold")
    
    for idx, col in enumerate(customers.columns):
        row, col_idx = divmod(idx, 3)
        axes[row, col_idx].hist(customers[col], bins=25, color="#3498db", 
                                 edgecolor="white", alpha=0.8)
        axes[row, col_idx].set_title(col)
        axes[row, col_idx].axvline(customers[col].mean(), color="red", linestyle="--",
                                    label=f"Mean: {customers[col].mean():.1f}")
        axes[row, col_idx].legend(fontsize=8)
    
    # Use the last subplot for pairwise scatter
    axes[1, 2].scatter(customers["Annual_Income_K"], customers["Spending_Score"],
                        alpha=0.5, s=20, color="#2ecc71")
    axes[1, 2].set_xlabel("Annual Income ($K)")
    axes[1, 2].set_ylabel("Spending Score")
    axes[1, 2].set_title("Income vs Spending")
    
    plt.tight_layout()
    plt.savefig("../notebooks/eda_customers.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: Feature Scaling
    # ══════════════════════════════════════════════════════════
    print_header("STEP 3: Feature Scaling")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customers)
    
    print("  Before scaling — Feature ranges:")
    for i, col in enumerate(customers.columns):
        print(f"    {col:25s}: [{customers[col].min():>6.1f}, {customers[col].max():>6.1f}]")
    
    print("\n  After scaling — All features approximately [-3, 3]")
    X_scaled_df = pd.DataFrame(X_scaled, columns=customers.columns)
    for col in X_scaled_df.columns:
        print(f"    {col:25s}: [{X_scaled_df[col].min():>6.2f}, {X_scaled_df[col].max():>6.2f}]")
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: Find Optimal K
    # ══════════════════════════════════════════════════════════
    print_header("STEP 4: Finding Optimal K")
    
    K_range = range(2, 11)
    inertias = []
    sil_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)
        print(f"  K={k:2d}: Inertia={kmeans.inertia_:>10.1f} | Silhouette={sil:.4f}")
    
    optimal_k = list(K_range)[np.argmax(sil_scores)]
    print(f"\n  🏆 Optimal K = {optimal_k} (highest silhouette: {max(sil_scores):.4f})")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimal K Selection", fontsize=16, fontweight="bold")
    
    ax1.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia")
    ax1.axvline(optimal_k, color="red", linestyle="--", alpha=0.7)
    
    ax2.bar(K_range, sil_scores, color="#2ecc71", edgecolor="white")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Score")
    ax2.axvline(optimal_k, color="red", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("../notebooks/optimal_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: Final K-Means Clustering
    # ══════════════════════════════════════════════════════════
    print_header("STEP 5: K-Means Clustering")
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    customers["Cluster"] = kmeans_final.fit_predict(X_scaled)
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: Segment Profiling
    # ══════════════════════════════════════════════════════════
    print_header("STEP 6: Segment Profiles")
    
    segment_names = {}
    for cluster in range(optimal_k):
        segment = customers[customers["Cluster"] == cluster]
        avg_income = segment["Annual_Income_K"].mean()
        avg_spending = segment["Spending_Score"].mean()
        avg_visits = segment["Monthly_Visits"].mean()
        
        # Auto-generate segment names based on characteristics
        if avg_income > 70 and avg_spending > 60:
            name = "💎 VIP High-Value"
        elif avg_income > 70 and avg_spending < 40:
            name = "💰 Premium Cautious"
        elif avg_income < 35 and avg_spending > 60:
            name = "🛍️ Young Enthusiasts"
        elif avg_income < 35 and avg_spending < 30:
            name = "📊 Budget Shoppers"
        else:
            name = "🎯 Average Customers"
        
        segment_names[cluster] = name
        
        print(f"\n  Cluster {cluster}: {name}")
        print(f"  {'─' * 50}")
        print(f"  Size:              {len(segment):>6} customers ({len(segment)/len(customers):.0%})")
        print(f"  Avg Age:           {segment['Age'].mean():>6.1f} years")
        print(f"  Avg Income:        ${avg_income:>5.1f}K / year")
        print(f"  Avg Spending Score:{avg_spending:>6.1f} / 100")
        print(f"  Avg Monthly Visits:{avg_visits:>6.1f}")
        print(f"  Avg Purchase Value:${segment['Avg_Purchase_Value'].mean():>5.0f}")
        
        # Marketing recommendation
        if "VIP" in name:
            print(f"  📌 Strategy: Exclusive offers, loyalty rewards, VIP events")
        elif "Cautious" in name:
            print(f"  📌 Strategy: Premium quality messaging, value propositions")
        elif "Enthusiasts" in name:
            print(f"  📌 Strategy: Trendy products, social media marketing, discounts")
        elif "Budget" in name:
            print(f"  📌 Strategy: Value deals, bundled offers, email coupons")
        else:
            print(f"  📌 Strategy: General promotions, seasonal campaigns")
    
    # ══════════════════════════════════════════════════════════
    # STEP 7: Final Visualization Dashboard
    # ══════════════════════════════════════════════════════════
    print_header("STEP 7: Visualization Dashboard")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Customer Segmentation Dashboard", fontsize=18, fontweight="bold")
    
    cmap = plt.cm.get_cmap("viridis", optimal_k)
    
    # Plot 1: Income vs Spending
    scatter = axes[0, 0].scatter(
        customers["Annual_Income_K"], customers["Spending_Score"],
        c=customers["Cluster"], cmap="viridis", alpha=0.6, s=40, edgecolors="white"
    )
    centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
    axes[0, 0].scatter(centers_original[:, 1], centers_original[:, 2],
                        marker="X", s=200, c="red", edgecolors="black", linewidth=2)
    axes[0, 0].set_xlabel("Annual Income ($K)")
    axes[0, 0].set_ylabel("Spending Score")
    axes[0, 0].set_title("Income vs Spending by Segment")
    
    # Plot 2: Segment sizes (pie chart)
    sizes = customers["Cluster"].value_counts().sort_index()
    labels = [segment_names.get(i, f"Cluster {i}") for i in sizes.index]
    colors = [cmap(i / optimal_k) for i in range(optimal_k)]
    axes[0, 1].pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors,
                    startangle=90, textprops={"fontsize": 9})
    axes[0, 1].set_title("Segment Distribution")
    
    # Plot 3: Radar / box comparison
    cluster_means = customers.groupby("Cluster")[
        ["Age", "Annual_Income_K", "Spending_Score", "Monthly_Visits"]
    ].mean()
    cluster_means.T.plot(kind="bar", ax=axes[1, 0], colormap="viridis", edgecolor="white")
    axes[1, 0].set_title("Segment Feature Comparison")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=30, ha="right")
    axes[1, 0].legend(title="Cluster", loc="upper right", fontsize=8)
    
    # Plot 4: Silhouette plot
    sample_silhouette_values = silhouette_samples(X_scaled, customers["Cluster"])
    y_lower = 10
    for i in range(optimal_k):
        ith_cluster = sample_silhouette_values[customers["Cluster"] == i]
        ith_cluster.sort()
        size_i = ith_cluster.shape[0]
        y_upper = y_lower + size_i
        color = cmap(i / optimal_k)
        axes[1, 1].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster,
                                   facecolor=color, edgecolor=color, alpha=0.7)
        axes[1, 1].text(-0.05, y_lower + 0.5 * size_i, str(i), fontsize=10)
        y_lower = y_upper + 10
    
    avg_sil = silhouette_score(X_scaled, customers["Cluster"])
    axes[1, 1].axvline(avg_sil, color="red", linestyle="--",
                        label=f"Avg: {avg_sil:.3f}")
    axes[1, 1].set_title("Silhouette Plot")
    axes[1, 1].set_xlabel("Silhouette Coefficient")
    axes[1, 1].set_ylabel("Cluster")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("../notebooks/segmentation_dashboard.png", dpi=150, bbox_inches="tight")
    print("  Saved: segmentation_dashboard.png")
    plt.show()
    
    print("\n✅ Customer Segmentation Complete!")
    print("   → Use these segments for targeted marketing campaigns")
    print("   → Next: Compare with DBSCAN (02_dbscan_anomaly_detection.py)")


if __name__ == "__main__":
    main()
