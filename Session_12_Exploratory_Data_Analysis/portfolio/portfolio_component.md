# 📊 Portfolio Component — Session 12

## Assignment: Automated EDA Dashboard (Streamlit)

### Overview
Exploratory Data Analysis (EDA) is time-consuming. In the real world, Data Scientists often build automated tools so they (or their non-technical managers) can instantly visualize new data without writing code.

For this portfolio project, you will build an **Automated Data Quality Dashboard** using `Streamlit`. You will deploy this dashboard to Hugging Face Spaces. This project proves you can engineer full-stack data tools that identify patterns and anomalies interactively.

---

### Deliverables

| # | Deliverable | Format | Estimated Time |
|---|------------|--------|----------------|
| 1 | Automated EDA App | Python (`app.py` via Streamlit) | 2-3 hours |
| 2 | Implementation Report | Markdown (`README.md`) | 1 hour |

---

### Dashboard Requirements

#### Core Features (Required — 70 points)
1. **File Uploader** (10 pts)
   - Allow the user to upload any `.csv` file. 
2. **Missing Values Analyzer** (20 pts)
   - Automatically display a bar chart showing the percentage of missing values for every column.
   - Include a button to "Impute All NaNs with Median/Mode".
3. **Correlation Heatmap** (20 pts)
   - Filter the uploaded dataset for numerical columns automatically.
   - Display a responsive `Seaborn` or `Plotly` correlation heatmap to identify trends.
4. **Outlier Detector** (20 pts)
   - Allow the user to select a specific numerical column from a dropdown.
   - Display a Boxplot of that column, and calculate exactly how many outliers exist using the IQR method.

#### Advanced Features (Bonus — 30 points)
5. **Interactive Data Cleaning** (15 pts)
   - After the user clicks "Remove Outliers", display the *new* Boxplot side-by-side with the old one to show the "Refined Presentation" in action.
6. **Pandas Profiling Integration** (15 pts)
   - Embed an interactive `ydata-profiling` (formerly Pandas Profiling) HTML report directly inside your Streamlit app using `streamlit_pandas_profiling`.

---

### Implementation Guide (Streamlit Basics)

Instead of Gradio (which we used for model deployment), `Streamlit` is the absolute industry standard for building Data Dashboards.

**Starter App Code (`app.py`):**
```python
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Auto EDA Profiler", layout="wide")
st.title("🔍 Automated Exploratory Data Analysis")

# 1. File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(df.head())
    
    # 2. Missing Values Analysis
    st.write("### Missing Values Report")
    missing_data = df.isnull().sum()
    st.bar_chart(missing_data[missing_data > 0])
    
    # 3. Correlation Heatmap
    st.write("### Correlation Heatmap (Underlying Patterns)")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for correlation!")
        
    # 4. Interactive Outlier Detection
    st.write("### Irregularity Detection")
    target_col = st.selectbox("Select a column to check for outliers:", numeric_df.columns)
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x=target_col, ax=ax2, color="orange")
    st.pyplot(fig2)
```

### Deployment Instructions (Hugging Face Spaces)
1. Create a new Space and select **Streamlit** as the SDK.
2. In your `requirements.txt`, ensure you have:
   ```text
   streamlit
   pandas
   seaborn
   matplotlib
   # ydata-profiling (If attempting bonus)
   ```
3. Commit and push your files.

> 💡 **Why This Matters**: Showing that you can take raw, dirty data and instantly build a web interface that finds trends and outliers is one of the fastest ways to prove value as a Data Scientist or Analyst.
