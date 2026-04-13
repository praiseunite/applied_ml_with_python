# Session 11: Practice 2 Capstone Project 🏭

**Module Alignment:** TL11 (Try It Yourself: Chapters 4–6)  
**Topics Covered:** Imbalanced Data, Time Series Context, Ensemble Learning, Model Evaluation

Welcome to your second major Portfolio Project! In this capstone, you will synthesize the advanced techniques you learned in Sessions 7, 8, 9, and 10 to build an **Industrial IoT Predictive Maintenance System**.

---

## 📖 The Scenario

You have been hired as a Machine Learning Engineer by a major manufacturing company. They operate thousands of heavy milling machines equipped with IoT sensors that measure Tool Wear, Rotational Speed, Torque, and Air/Process Temperatures.

**The Problem:** When machines suddenly fail on the assembly line, it costs the company $50,000 per hour in downtime. 
**Your Goal:** Build a Machine Learning pipeline that monitors the time-series drift of these sensors and uses **Ensemble Learning** algorithms to predict an imminent failure *before* the machine breaks.

**The Catch (Imbalanced Data):** Machine failures are extremely rare. Out of 10,000 data points, maybe only 300 represent failures. If you don't handle this imbalance (using techniques like **SMOTE**), your model will be functionally useless!

---

## 🛠️ Project Structure

This capstone contains everything you need to build, evaluate, and deploy your project.

### 1. `data/`
Inside this folder, you will find `fetch_dataset.py`. We are using the very famous **AI4I 2020 Predictive Maintenance Dataset** (Open Source via the UCI Machine Learning Repository). Run the script to securely download and prepare the 10,000-row real-world dataset.

### 2. `notebooks/`
Open `Practice_02_Guided_Project.ipynb`. This notebook provides structural scaffolding to guide you through:
1. Feature Engineering & Time Series feature extraction.
2. Handling the severe Class Imbalance.
3. Training **Random Forest** and **XGBoost** classifiers.
4. Evaluating them using **ROC-AUC** and **Precision-Recall** trade-offs instead of naive accuracy.

### 3. `solutions/`
Stuck? A complete Master Reference Notebook (`Complete_Capstone_Solution.ipynb`) is provided by your instructor with all code fully implemented and documented.

### 4. `app/`
Once your model is trained and saved, you will build a production-ready Web Dashboard using `Gradio`. Users will be able to input live sensor readings and instantly see if the machine needs preventative maintenance!

---

## 🚀 Getting Started

1. Ensure your Virtual Environment is activated from the root `setup.bat / setup.sh`.
2. Navigate to the `data/` directory and run the data fetcher:
   ```bash
   python fetch_dataset.py
   ```
3. Open the `notebooks/` directory and begin the guided challenge!

---
*© 2024 Aptech Limited — For Educational Use*
