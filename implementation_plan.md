# Applied Machine Learning Using Python — World-Class Course Materials

## Goal

Create **production-grade, portfolio-ready training materials** for the 20-session (40-hour) AMLP course that serve beginners, intermediate, and advanced learners equally well, using real-world datasets, verified code, and industry best practices.

---

## Course Architecture Overview

```
c:\Projects\Aptech\Applied_ML_with_python\
│
├── README.md                          # Course overview & setup guide
├── requirements.txt                   # All Python dependencies
├── setup_guide.md                     # Environment setup (Anaconda, VS Code, Jupyter)
│
├── Session_01_Introduction_to_ML/
│   ├── notes/
│   │   └── Session_01_Notes.md        # Theory (Beginner → Advanced tiers)
│   ├── code/
│   │   ├── 01_ml_fundamentals.py      # Clean, documented scripts
│   │   └── 01_ml_fundamentals.ipynb   # Jupyter notebook version
│   ├── exercises/
│   │   ├── beginner_exercises.md
│   │   ├── intermediate_exercises.md
│   │   └── advanced_exercises.md
│   ├── solutions/
│   │   └── solutions.ipynb
│   └── portfolio/
│       └── portfolio_component.md     # What to add to portfolio from this session
│
├── Session_02_Advanced_Clustering/
│   └── ... (same structure)
│
├── ... (Sessions 03–12)
│
├── Practice_Sessions/
│   ├── Practice_01_Sessions_1_to_3/
│   ├── Practice_02_Sessions_4_to_6/
│   ├── Practice_03_Sessions_7_to_9/
│   └── Practice_04_Sessions_10_to_12/
│
├── Assignments/
│   ├── Assignment_Block_1_Sessions_1_to_6/
│   └── Assignment_Block_2_Sessions_7_to_12/
│
├── Portfolio_Projects/
│   ├── Project_01_Customer_Segmentation/        # Clustering + EDA
│   ├── Project_02_Stock_Price_Forecasting/       # Time Series + LSTM
│   ├── Project_03_Credit_Risk_Assessment/        # Imbalanced Data + Ensemble
│   ├── Project_04_RL_Game_Agent/                 # Reinforcement Learning
│   ├── Project_05_ML_Pipeline_Deployment/        # Flask API + Cloud
│   └── Project_06_AB_Testing_Platform/           # Causal Inference + A/B Testing
│
├── Datasets/
│   └── README.md                      # Dataset sources & descriptions
│
└── Cheatsheets/
    ├── sklearn_cheatsheet.md
    ├── pandas_cheatsheet.md
    └── ml_algorithms_cheatsheet.md
```

---

## User Review Required

> [!IMPORTANT]
> **Scope Decision**: This is a massive undertaking (12 content sessions + 4 practice sessions + 6 portfolio projects + assignments). I recommend we build this **incrementally by session blocks** rather than all at once. Proposed order:
> - **Phase 1**: Sessions 1–3 + Practice 1 + Portfolio Projects 1 & 4 (Week 1-2 coverage)
> - **Phase 2**: Sessions 4–6 + Practice 2 + Assignment Block 1 + Portfolio Projects 2 & 3 (Week 2-3)
> - **Phase 3**: Sessions 7–9 + Practice 3 + Portfolio Project 5 & 6 (Week 3-4)
> - **Phase 4**: Sessions 10–12 + Practice 4 + Assignment Block 2 (Week 5)
> - **Phase 5**: Final polish, cheatsheets, README, setup guide

> [!WARNING]
> **Real Datasets Only**: All code will use publicly available, well-known datasets (UCI ML Repository, Kaggle, scikit-learn built-ins, Yahoo Finance API, etc.). No synthetic/fake data for portfolio projects.

> [!IMPORTANT]
> **Code Framework**: All code will target **Python 3.10+** with these core libraries:
> - scikit-learn, pandas, numpy, matplotlib, seaborn
> - tensorflow/keras (for LSTM), prophet (for time series)
> - flask (for deployment), imbalanced-learn (for SMOTE)
> - gym/gymnasium (for RL), shap/lime (for explainability)
> - auto-sklearn or TPOT (for AutoML)

---

## Proposed Changes — Session-by-Session Breakdown

### Session 1: Introduction to Machine Learning (TL1 + TL2 = 4 hours)

**Covers**: Purpose of ML, history, ethical considerations, bias & fairness, real-world applications

| Level | Content |
|-------|---------|
| **Beginner** | What is ML? Supervised vs Unsupervised vs Reinforcement. Simple linear regression from scratch. Real examples (spam filter, Netflix recommendations) |
| **Intermediate** | ML pipeline overview. Bias-variance tradeoff intro. Scikit-learn API patterns. Cross-validation basics |
| **Advanced** | Ethical AI frameworks (EU AI Act, IEEE guidelines). Fairness metrics (demographic parity, equalized odds). Bias audit on real dataset |

**Real-World Practical**: Bias audit on the **Adult Census Income dataset** (UCI) — detect gender/race bias in income prediction
**Code Deliverables**:
- `01a_ml_from_scratch.py` — Linear regression without libraries
- `01b_sklearn_pipeline.py` — Full sklearn pipeline with preprocessing
- `01c_bias_audit.py` — Fairness analysis with AIF360/custom metrics

**Portfolio Component**: "Responsible AI Audit Report" — students analyze a real dataset for bias

---

### Session 2: Unsupervised Advanced Clustering Algorithms (TL3 = 2 hours)

**Covers**: Clustering definition, techniques (K-Means, DBSCAN, Hierarchical, Gaussian Mixture), dimensionality reduction

| Level | Content |
|-------|---------|
| **Beginner** | K-Means intuition, elbow method, silhouette scores. Visualizing clusters in 2D |
| **Intermediate** | DBSCAN for non-spherical clusters. Hierarchical clustering with dendrograms. Choosing the right algorithm |
| **Advanced** | Gaussian Mixture Models (EM algorithm). t-SNE vs UMAP for visualization. Spectral clustering |

**Real-World Practical**: **Customer Segmentation** on the Mall Customers dataset + **Image compression** using K-Means on actual images
**Code Deliverables**:
- `02a_kmeans_customer_segmentation.py`
- `02b_dbscan_anomaly_detection.py`
- `02c_dimensionality_reduction_comparison.py` (PCA vs t-SNE vs UMAP)

**Portfolio Component**: Customer Segmentation Dashboard (interactive Plotly visualizations)

---

### Session 3: Markov Decision Process & Reinforcement Learning (TL4 + TL5 = 4 hours)

**Covers**: MDP formulation, RL algorithms (Q-Learning, SARSA), Policy Gradient, Actor-Critic

| Level | Content |
|-------|---------|
| **Beginner** | What is RL? Agent-Environment loop. Grid world example. Q-table from scratch |
| **Intermediate** | Q-Learning with OpenAI Gymnasium (CartPole, FrozenLake). Epsilon-greedy exploration. SARSA comparison |
| **Advanced** | Deep Q-Networks (DQN). Policy Gradient (REINFORCE). Actor-Critic (A2C). PPO overview |

**Real-World Practical**: Train an agent to play **CartPole** and **Taxi-v3** using OpenAI Gymnasium
**Code Deliverables**:
- `03a_gridworld_qtable.py` — Q-learning from scratch
- `03b_cartpole_qlearning.py` — Q-learning with Gymnasium
- `03c_policy_gradient_cartpole.py` — REINFORCE algorithm
- `03d_actor_critic.py` — A2C implementation

**Portfolio Component**: RL Game Agent with training curves and performance analysis

---

### Session 4: Handling Imbalanced Data (TL7 = 2 hours)

**Covers**: Feature selection, PCA, SMOTE, oversampling/undersampling

| Level | Content |
|-------|---------|
| **Beginner** | Why class imbalance matters. Random oversampling/undersampling. Confusion matrix deep dive |
| **Intermediate** | SMOTE and its variants (BorderlineSMOTE, ADASYN). Feature selection (filter, wrapper, embedded methods). PCA step-by-step |
| **Advanced** | Cost-sensitive learning. Ensemble methods for imbalanced data (BalancedRandomForest, EasyEnsemble). Feature importance with SHAP |

**Real-World Practical**: **Credit Card Fraud Detection** using the Kaggle Credit Card Fraud dataset (284,807 transactions, 492 frauds)
**Code Deliverables**:
- `04a_imbalanced_basics.py` — Visualizing imbalance, basic resampling
- `04b_smote_variants.py` — SMOTE, BorderlineSMOTE, ADASYN comparison
- `04c_feature_selection_pca.py` — SelectKBest, RFE, PCA pipeline
- `04d_fraud_detection_pipeline.py` — Complete fraud detection system

**Portfolio Component**: Credit Card Fraud Detection System with ROC analysis

---

### Session 5: Time Series Data Pre-processing (TL8 = 2 hours)

**Covers**: Time series models, Prophet, LSTM implementation

| Level | Content |
|-------|---------|
| **Beginner** | What is time series? Trend, seasonality, stationarity. Moving averages. ADF test |
| **Intermediate** | ARIMA/SARIMA modeling. Facebook Prophet for forecasting. Decomposition |
| **Advanced** | LSTM networks for sequence prediction. Bidirectional LSTM. Multi-step forecasting |

**Real-World Practical**: **Stock Price Forecasting** (Yahoo Finance API) + **Energy Demand Prediction** (UCI Household Power Consumption)
**Code Deliverables**:
- `05a_time_series_basics.py` — Decomposition, stationarity tests
- `05b_arima_forecasting.py` — ARIMA/SARIMA on real data
- `05c_prophet_forecasting.py` — Prophet for stock prices
- `05d_lstm_prediction.py` — LSTM for multi-step forecasting

**Portfolio Component**: Stock Price Forecasting Dashboard with multiple model comparison

---

### Session 6: Ensemble Learning & Model Evaluation (TL9 + TL10 = 4 hours)

**Covers**: Ensemble techniques, voting classifiers/regressors, cross-validation, bias-variance, ROC curves

| Level | Content |
|-------|---------|
| **Beginner** | Bagging vs Boosting intuition. Random Forest. Voting classifiers. Accuracy vs Precision vs Recall |
| **Intermediate** | Gradient Boosting, XGBoost, LightGBM. Stratified K-Fold CV. AUC-ROC analysis. Hyperparameter tuning (GridSearch, RandomSearch) |
| **Advanced** | Stacking ensembles. Bayesian Optimization for hyperparameters. Statistical significance testing of models. Learning curves analysis |

**Real-World Practical**: **Heart Disease Prediction** using the UCI Heart Disease dataset — compare 10+ models
**Code Deliverables**:
- `06a_ensemble_basics.py` — Bagging, boosting, voting
- `06b_xgboost_lightgbm.py` — Advanced boosting with tuning
- `06c_cross_validation_deep_dive.py` — CV strategies comparison
- `06d_model_evaluation_complete.py` — ROC, PR curves, confusion matrices, statistical tests

**Portfolio Component**: Heart Disease Prediction System with comprehensive model comparison report

---

### Session 7: Exploratory Data Analysis (TL12 + TL13 = 4 hours)

**Covers**: EDA role, missing values, central tendency, dispersion, advanced EDA techniques

| Level | Content |
|-------|---------|
| **Beginner** | Descriptive statistics. Histograms, box plots, scatter plots. Handling missing values (mean, median, mode) |
| **Intermediate** | Correlation analysis. Outlier detection (IQR, Z-score). Data profiling with pandas-profiling. Feature engineering basics |
| **Advanced** | Advanced imputation (KNN, MICE). Multivariate analysis. Automated EDA (Sweetviz, D-Tale). Statistical testing for distributions |

**Real-World Practical**: **Titanic Survival Analysis** + **House Prices EDA** (Kaggle datasets)
**Code Deliverables**:
- `07a_eda_fundamentals.py` — Descriptive stats, basic plots
- `07b_missing_values_strategies.py` — Imputation comparison
- `07c_advanced_visualization.py` — Seaborn/Plotly advanced charts
- `07d_automated_eda.py` — pandas-profiling, Sweetviz reports

**Portfolio Component**: Comprehensive EDA Report with interactive visualizations

---

### Session 8: Model Deployment & Maintenance (TL14 = 2 hours)

**Covers**: Serialization (pickle/joblib), Flask APIs, cloud deployment

| Level | Content |
|-------|---------|
| **Beginner** | Saving/loading models with pickle and joblib. What is an API? Simple Flask hello world |
| **Intermediate** | RESTful API design. Flask ML prediction endpoint. Request/response handling. Input validation |
| **Advanced** | Docker containerization. CI/CD pipeline concepts. Model monitoring. A/B testing deployment. Cloud deployment (Render/Railway) |

**Real-World Practical**: Deploy a trained ML model as a **Flask REST API** with a frontend, then deploy to **Render** (free tier)
**Code Deliverables**:
- `08a_model_serialization.py` — Save/load models
- `08b_flask_api/` — Complete Flask API project structure
- `08c_deployment/` — Dockerfile, requirements, deployment configs
- `08d_model_monitoring.py` — Drift detection basics

**Portfolio Component**: Deployed ML API with live URL (students deploy their own)

---

### Session 9: Causal Inference & A/B Testing (TL15 = 2 hours)

**Covers**: Causal inference, methods, A/B testing process, propensity score matching

| Level | Content |
|-------|---------|
| **Beginner** | Correlation ≠ Causation. What is A/B testing? Control vs treatment groups. P-values explained simply |
| **Intermediate** | Hypothesis testing. Sample size calculation. A/B test analysis. Propensity Score Matching basics |
| **Advanced** | Instrumental variables. Difference-in-differences. Regression discontinuity. Causal forests. DoWhy library |

**Real-World Practical**: **E-commerce A/B Test Analysis** — analyze whether a new checkout page increases conversion
**Code Deliverables**:
- `09a_ab_testing_basics.py` — Hypothesis tests, p-values, effect sizes
- `09b_sample_size_calculator.py` — Power analysis
- `09c_propensity_score_matching.py` — PSM implementation
- `09d_causal_inference_dowhy.py` — DoWhy causal analysis

**Portfolio Component**: A/B Testing Analysis Report (simulated e-commerce experiment)

---

### Session 10: AutoML & Transparent Models (TL17 = 2 hours)

**Covers**: AutoML significance, automated models, model transparency

| Level | Content |
|-------|---------|
| **Beginner** | What is AutoML? Why it matters. Using auto-sklearn/TPOT for automatic model selection |
| **Intermediate** | H2O AutoML. Feature importances. LIME for local explanations. Partial dependence plots |
| **Advanced** | SHAP values deep dive. Anchor explanations. Model cards. Interpretable models (GAMs, rule lists) |

**Real-World Practical**: Compare AutoML vs manual ML on the **Diabetes dataset** + Explain a black-box model with SHAP/LIME
**Code Deliverables**:
- `10a_automl_tpot.py` — Automated ML pipeline
- `10b_h2o_automl.py` — H2O AutoML
- `10c_shap_explanations.py` — SHAP analysis
- `10d_lime_explanations.py` — LIME local explanations

**Portfolio Component**: Model Explainability Report with SHAP/LIME visualizations

---

### Session 11: Ethical Considerations in AI (TL18 = 2 hours)

**Covers**: Bias types, responsible AI practices

| Level | Content |
|-------|---------|
| **Beginner** | Types of bias (selection, confirmation, automation). Famous AI failures (Amazon hiring, COMPAS) |
| **Intermediate** | Fairness metrics implementation. Bias mitigation strategies. AI governance frameworks |
| **Advanced** | Algorithmic auditing. Differential privacy. Federated approach to ethics. Creating model cards & datasheets |

**Real-World Practical**: **Algorithmic Fairness Audit** on the COMPAS recidivism dataset
**Code Deliverables**:
- `11a_bias_detection.py` — Detect bias in datasets and models
- `11b_fairness_metrics.py` — Implement demographic parity, equalized odds
- `11c_bias_mitigation.py` — Pre/in/post-processing mitigation
- `11d_model_card_generator.py` — Automated model card creation

**Portfolio Component**: AI Ethics Case Study & Bias Audit Report

---

### Session 12: Emerging Trends in AI & ML (TL19 = 2 hours)

**Covers**: Emerging trends, XAI, Federated Learning, key trends

| Level | Content |
|-------|---------|
| **Beginner** | Current AI landscape. LLMs overview. AI in healthcare, finance, climate. Career paths in AI |
| **Intermediate** | Explainable AI (XAI) techniques recap. Federated Learning concept and implementation. Edge AI |
| **Advanced** | Federated Learning with PySyft. Neural Architecture Search. AI safety research. Multi-modal models |

**Real-World Practical**: **Federated Learning simulation** + **XAI comparison study**
**Code Deliverables**:
- `12a_xai_techniques_comparison.py` — Compare XAI methods
- `12b_federated_learning_simulation.py` — Simulated FL with PySyft/Flower
- `12c_future_trends_demo.py` — Interactive demos of cutting-edge techniques

**Portfolio Component**: AI Trends Research Paper / Blog Post

---

### Practice Sessions (TL6, TL11, TL16, TL20)

Each practice session includes:
- **5–8 "Try It Yourself" problems** per session block (graded by difficulty: ⭐ Beginner, ⭐⭐ Intermediate, ⭐⭐⭐ Advanced)
- **Timed coding challenges** (30-min mini-projects)
- **Peer review rubrics** for code quality assessment
- **Complete solutions** with detailed explanations

---

### Portfolio Projects (6 Capstone Projects)

| # | Project | Sessions Used | Real Dataset | Deployment |
|---|---------|--------------|--------------|------------|
| 1 | Customer Segmentation Engine | 2, 7 | Mall Customers + E-commerce | Streamlit Dashboard |
| 2 | Stock Price Forecasting System | 5, 6 | Yahoo Finance API | Flask API |
| 3 | Credit Risk Assessment Platform | 4, 6 | Kaggle Credit/Lending Club | Flask + HTML Frontend |
| 4 | RL Game Agent | 3 | OpenAI Gymnasium | Recorded demo video |
| 5 | End-to-End ML Pipeline | 7, 8, 10 | Any Kaggle dataset | Deployed on Render |
| 6 | A/B Testing Analytics Platform | 9, 11 | Simulated e-commerce | Jupyter Report |

Each project includes:
- `README.md` with project overview and screenshots
- Clean, documented code
- Requirements file
- Deployment instructions
- Results/analysis report

---

## Content Quality Standards

### Code Standards
- **Every script** has docstrings, type hints, and comments
- **PEP 8** compliant
- **Error handling** included
- **Logging** for production code
- **Unit tests** for critical functions
- **Requirements pinned** to specific versions

### Theory Standards
- **Real citations** from published papers and textbooks
- **Mathematical formulas** with plain-English explanations
- **Visual diagrams** for every algorithm
- **"Why This Matters"** sections connecting to industry use
- **Common Pitfalls** sections to prevent typical mistakes

### Multi-Level Approach
Each session note follows this structure:
```
## 🎯 Learning Objectives

## 📋 Prerequisites

## 🟢 Beginner Level
   - Core concepts with analogies
   - Step-by-step walkthrough
   - Simple code examples

## 🟡 Intermediate Level
   - Deeper mathematical intuition
   - Real-world implementation patterns
   - Performance optimization

## 🔴 Advanced Level
   - Research paper references
   - Edge cases and limitations
   - Industry-scale considerations

## 💻 Hands-On Lab
   - Guided exercises
   - Challenge problems

## 📊 Portfolio Task
   - What to add to your portfolio

## 📚 Further Reading
```

---

## Open Questions

> [!IMPORTANT]
> 1. **Should I start building Phase 1 (Sessions 1–3) now**, or do you want to review/modify this plan first?
> 2. **Do you have a preferred Python version?** I'm planning for Python 3.10+
> 3. **Cloud deployment target**: Render (free), Railway, or Heroku? Or should I cover multiple?
> 4. **Assessment format**: Do you want MCQ quizzes alongside the practicals, or only coding exercises?
> 5. **Presentation slides**: Do you also need PowerPoint/Google Slides, or just markdown notes + Jupyter notebooks?
> 6. **Student level baseline**: Can I assume students already know basic Python and data science fundamentals (pandas, numpy, matplotlib)?

---

## Verification Plan

### Automated Tests
- Every Python script will be **tested by running it** before delivery
- All notebooks will be **executed end-to-end** to verify outputs
- Dataset downloads will be verified for availability
- Package compatibility will be tested with `pip install -r requirements.txt`

### Manual Verification
- Cross-reference all theory against the 3 reference books:
  - *Machine Learning with Python Cookbook* (Chris Albon)
  - *Hands-On ML with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron)
  - *Python Machine Learning* (Sebastian Raschka & Vahid Mirjalili)
- Code output screenshots included in notebooks
- Portfolio project deployment URLs tested

### Quality Checklist per Session
- [ ] Theory notes complete (all 3 levels)
- [ ] Code runs without errors
- [ ] Real dataset used (not synthetic)
- [ ] Exercises provided (3 difficulty levels)
- [ ] Solutions provided and tested
- [ ] Portfolio component defined
- [ ] Further reading links verified
