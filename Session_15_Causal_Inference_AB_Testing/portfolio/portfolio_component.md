# 📊 Portfolio Component — Session 15

## Assignment: A/B Testing Statistical Platform

### Overview
This is **Portfolio Project #6**. Proving causality is the most valuable skill a Data Scientist can offer a business. 

For this project, you will build a complete **A/B Testing Dashboard** using `Streamlit`. The Marketing or Product team should be able to upload a spreadsheet of conversion data, and your ML application will mathematically instantly tell them if their new feature is a success or a failure.

---

### Deliverables

| # | Deliverable | Format | Estimated Time |
|---|------------|--------|----------------|
| 1 | A/B Test Dashboard | Python (`app.py` via Streamlit) | 2.5 hours |
| 2 | Live Deployment | Hugging Face Spaces | 15 mins |

---

### Project Requirements

#### Core Specifications (100 points)

**1. File Ingestion (20 pts)**
Create an interface where a user can upload a CSV file containing 2 columns:
- `Group`: Contains values "A" (Control) and "B" (Treatment).
- `Converted`: Contains binary values `0` (Did not click) and `1` (Clicked/Bought).

**2. Metric Extraction (30 pts)**
Use standard Pandas to extract:
- Total size of Group A vs Group B.
- The raw Conversion Rate for Group A vs Group B (e.g. 10.5% vs 12.1%).
- Display these as big, bold `st.metric()` or `st.columns()` in Streamlit.

**3. The Causality Engine (30 pts)**
- Implement `scipy.stats.ttest_ind`.
- Extract the specific P-Value.
- If `P < 0.05`, display a giant Green success message: "✅ STATISTICALLY SIGNIFICANT. The Treatment Caused the Uplift!"
- If `P >= 0.05`, display a giant Red failure message: "❌ NOT SIGNIFICANT. The difference is just statistical noise. Do not launch."

**4. Visual Distribution (20 pts)**
- Generate a Seaborn Bar chart comparing the conversion rates between A and B, so non-technical stakeholders visually understand the difference your numbers are proving.
- Embed the plot using `st.pyplot()`.

---

### Starter Data (Mock User Data)
If you need data to test your app during development, use this script to generate `ab_test_data.csv`:

```python
import numpy as np
import pandas as pd

np.random.seed(42)
group_a = np.random.binomial(1, 0.08, 2000) # Control: 8% CR
group_b = np.random.binomial(1, 0.11, 2000) # Treatment: 11% CR

df = pd.DataFrame({
    'Group': ['A'] * 2000 + ['B'] * 2000,
    'Converted': np.concatenate([group_a, group_b])
})
df.to_csv('ab_test_data.csv', index=False)
```

> 💡 **Why This Matters**: Every major tech company in the world (Amazon, Netflix, Meta) runs completely on A/B tests. Having a live, interactive deployed A/B Testing platform in your portfolio proves you understand massive enterprise decision-making flow.
