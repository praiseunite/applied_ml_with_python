# Session 16: Practice 3 Capstone Project 🚀

**Module Alignment:** TL16 (Try It Yourself: Chapters 7–9)  
**Topics Covered:** Exploratory Data Analysis, Model Deployment & APIs, Causal Inference, A/B Testing

Welcome to your third major Portfolio Project! In this capstone, you will synthesize the incredibly powerful techniques you learned in Sessions 12, 13, 14, and 15 to build an **End-to-End A/B Testing Microservice**.

This project perfectly aligns with both **Project #5 (ML Pipeline Deployment)** and **Project #6 (A/B Testing Platform)** from the master course index!

---

## 📖 The Scenario

You have been hired as a full-stack Machine Learning Engineer at a major E-Commerce company. 
The marketing team just concluded a 2-week A/B test changing the website's checkout button from Blue (Group A) to Green (Group B).

They collected a massive file of raw conversion data, but it is **highly messy**:
- Sensors malfunctioned, producing `NaN` (Missing) values.
- "Whale" buyers (outliers spending $50,000 in one transaction) have skewed the data.

**Your Goal:** You must build an automated **Flask Web Server** that receives this raw data via HTTP POST, programmatically cleans the irregularities (EDA), runs a statistical T-Test on the cleaned data (Causality), and returns a JSON response deciding if the color change was a success. Finally, you will package it in a **Docker Container** for deployment.

---

## 🛠️ Project Structure

### 1. `data/`
Run the `generate_ab_test_data.py` script. It will generate `raw_ab_data.json` containing 5000 user transactions. Notice how it is intentionally peppered with outliers and missing values to test your EDA capabilities.

### 2. `notebooks/`
Open `Practice_03_Guided_Project.ipynb`. This notebook provides structural scaffolding. Use this as your "Sandbox" to write the data cleaning and T-test logic *before* you move it into the Flask Web Server.

### 3. `app/`
This is the final deployment folder.
- `app.py`: Your Flask API. It must contain an endpoint `@app.route('/analyze_ab_test')`.
- `test_api.py`: A script mimicking the marketing department sending data to your server.
- `Dockerfile`: The configuration needed to deploy your API to Hugging Face Spaces.

### 4. `solutions/`
Stuck? A complete Master Reference Notebook (`Complete_Capstone_Solution.ipynb`) is provided by your instructor with all cleaning pipelines and statistical math fully implemented.

---

## 🚀 Getting Started

1. Ensure your Virtual Environment is activated from the root `setup.bat / setup.sh`.
2. Navigate to the `data/` directory and run the data generator:
   ```bash
   python generate_ab_test_data.py
   ```
3. Open the `notebooks/` directory and write the logic!
4. Move your logic into `app/app.py` and start the server!

---
*© 2024 Aptech Limited — For Educational Use*
