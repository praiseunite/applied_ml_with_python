# Practice 3 Capstone — Coverage Checklist

## TL16 Topics

This is the Capstone project synthesizing Chapters 7 through 9.

| # | Chapter Reference | Required Implementation | Covered In | Status |
|---|-------------------|-------------------------|------------|--------|
| 1 | Chapter 7 (EDA) | Identify irregular outliers and missing missing values safely. | `app.py` `clean_data()` func. Eliminating generated JSON NaNs and defining dynamic IQR boundaries for whales. | ✅ |
| 2 | Chapter 8 (Deployment)| Formulate ML algorithms and statistics into Web APIs on Cloud architecture. | `app.py` Flask Endpoints, HTTP JSON parsing, and the `Dockerfile` specification. | ✅ |
| 3 | Chapter 9 (Causality)| Execute the A/B testing step-by-step process. | `app.py` `run_ab_test()` function executing the scipy t-test and returning standard p-values. | ✅ |

### Validation Note
We have successfully mapped **Portfolio Project 5** (End-To-End API Deployment via Docker) and **Portfolio Project 6** (A/B Testing Platform) into a singular, unified Capstone Application. The students now possess a production-ready repository where data pipelines trigger statistical engines across HTTP networks.
