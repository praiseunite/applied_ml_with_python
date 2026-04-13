# 📊 Portfolio Component — Session 13

## Assignment: Advanced Statistical Analysis Dashboard

### Overview
In Session 12, you built an Automated Data Profiler that found basic correlations and missing values. Now, you must upgrade that `Streamlit` dashboard into a full **Statistical Analysis Engine**.

When dealing with highly sensitive data (like financial audits or healthcare triage), standard averages are not enough. Your stakeholders need to know the *exact shape* of the data they are relying on.

---

### Deliverables

| # | Deliverable | Format | Estimated Time |
|---|------------|--------|----------------|
| 1 | Statistical Analysis App | Python (`app.py` via Streamlit) | 2 hours |
| 2 | Live Deployment | Hugging Face Spaces | 15 mins |

---

### Dashboard Upgrades

#### Required Upgrades (100 points)
Integrate the following 3 tabs into your existing Streamlit application.

**1. The Central Tendency Warning System (30 pts)**
Create an interface where a user selects a column.
- Calculate the `Mean` and the `Median`.
- Output a dynamic warning alert: If the `Mean` is significantly higher or lower than the `Median` (e.g., > 15% difference), flash a yellow warning Box: *"WARNING: Data is highly skewed. Use the Median for reporting, not the Mean!"*

**2. The Shape Engine (35 pts)**
Using `scipy.stats` or standard `pandas`:
- Calculate the exact **Skewness** and **Kurtosis** for the selected column.
- Print them out clearly. Add a text explainer next to Kurtosis telling the user what it implies (e.g. "Kurtosis is > 3: High risk of extreme outliers.").
- Display a KDE Distribution plot (`sns.histplot(kde=True)`) right below the stats so they can verify the mathematical shape visually.

**3. Advanced 3D Multivariate Plotting (35 pts)**
Real-world data is multi-dimensional. Bring in `Plotly` (`import plotly.express as px`) to build a 3D Scatter plot!
- Let the user select an X, Y, and Z axis variable from dropdowns.
- Let the user select a Category variable to "color" the dots.
- Render the 3D graph within Streamlit using `st.plotly_chart()`.

---

### Implementation Example (Plotly 3D inside Streamlit)

```python
import streamlit as st
import plotly.express as px
import pandas as pd

# Assume 'df' is loaded from your file uploader
st.write("### 🌐 Advanced 3D Multivariate Analysis")

# Setup 3 column layout for dropdowns
col1, col2, col3, col4 = st.columns(4)
with col1: x_var = st.selectbox("X-Axis", df.select_dtypes('number').columns)
with col2: y_var = st.selectbox("Y-Axis", df.select_dtypes('number').columns, index=1)
with col3: z_var = st.selectbox("Z-Axis", df.select_dtypes('number').columns, index=2)
with col4: color_var = st.selectbox("Color Category", df.select_dtypes('object').columns)

# Generate Plotly 3D Figure
fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var, color=color_var, opacity=0.7)
fig.update_traces(marker_size=5)

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)
```

> 💡 **Why This Matters**: Upgrading your basic dashboard with 3D renderings and mathematical safety checks (Skew/Kurtosis) transitions your app from a "toy" into a professional tool that actuaries and financial quants can trust.
