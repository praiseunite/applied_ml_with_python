# 📊 Portfolio Component — Session 02

## Assignment: Customer Segmentation Dashboard

### Overview

Build a **Customer Segmentation Analysis** that demonstrates your ability to apply multiple clustering algorithms, evaluate results, and derive actionable business insights.

---

### Deliverables

| # | Deliverable | Format | Time |
|---|------------|--------|------|
| 1 | Segmentation Analysis Report | Markdown `.md` | 2 hrs |
| 2 | Complete Analysis Code | `.py` + `.ipynb` | 2 hrs |
| 3 | Visualization Dashboard | PNG images | Included |

---

### Requirements

#### Data
Use any ONE of these real datasets:
- **Mall Customers** (Kaggle): `kaggle.com/datasets/vjchoudhary7/customer-segmentation`
- **Online Retail** (UCI): `archive.ics.uci.edu/dataset/352/online+retail`
- **Credit Card** (Kaggle): `kaggle.com/datasets/arjunbhasin2013/ccdata`
- Or generate realistic data using the code from `01_kmeans_customer_segmentation.py`

#### Analysis Steps

1. **Exploratory Data Analysis** — distributions, correlations, outliers
2. **Feature Engineering** — create RFM features (Recency, Frequency, Monetary) if applicable
3. **Apply 3+ Algorithms**:
   - K-Means (with elbow + silhouette for optimal K)
   - DBSCAN (with k-distance tuning)
   - Hierarchical (with dendrogram)
4. **Dimensionality Reduction** — PCA or t-SNE for visualization
5. **Segment Profiling** — name each segment, describe characteristics
6. **Business Recommendations** — actionable marketing strategy per segment

#### Report Structure

```markdown
# Customer Segmentation Report
## 1. Executive Summary
## 2. Dataset & EDA
## 3. Methodology (algorithms + parameters)
## 4. Results (with comparison table)
## 5. Segment Profiles (with visualizations)
## 6. Business Recommendations
## 7. Conclusion
```

---

### Grading Rubric

| Criteria | Points |
|----------|--------|
| EDA quality and depth | 15 |
| Algorithm implementation (3+) | 25 |
| Evaluation metrics and comparison | 20 |
| Visualizations | 15 |
| Business insight quality | 15 |
| Code quality and documentation | 10 |
| **Total** | **100** |

---

### Deployment (Session 8)

Later in the course, you'll deploy this as an **interactive Streamlit/Gradio app** on Hugging Face Spaces where users can:
- Upload their own customer data
- Choose clustering algorithm and parameters
- View cluster visualizations in real-time
- Download segment labels
