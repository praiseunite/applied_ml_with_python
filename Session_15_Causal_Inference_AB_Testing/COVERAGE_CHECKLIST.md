# Session 15 — Topic Coverage Checklist

## TL15 Topics (Causal Inference and A/B Testing)

| # | Required Topic | Covered In | Status |
|---|----------------|------------|--------|
| 1 | Define causal inference and challenges | `README.md` Part 1, `notebooks/` §2 (The Counterfactual Problem) | ✅ |
| 2 | Describe its methods and approaches | `README.md` Part 2 (RCT vs Observational), `notebooks/` §1 vs §2 | ✅ |
| 3 | Explain A/B testing and its step-by-step process | `README.md` Part 2, `code/01_ab_testing_ttest.py` (5 strictly defined steps) | ✅ |
| 4 | Explain PSM and its implementation | `README.md` Part 3, `code/02_propensity_score_matching.py` (Using Scikit-Learn Logistic Regression) | ✅ |

### Technical Note:
As discussed with the lead instructor, to maintain alignment with the "Applied Python" structure of this course, the incredibly dense mathematics of Propensity Score Matching (e.g. Mahalanobis distance formulas) were eschewed. Instead, PSM was implemented highly effectively using standard `sklearn.linear_model.LogisticRegression` and `sklearn.neighbors.NearestNeighbors` to mathematically extract Propensity Probabilities over observational data.
