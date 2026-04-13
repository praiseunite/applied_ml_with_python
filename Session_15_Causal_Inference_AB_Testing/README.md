# Session 15: Causal Inference and A/B Testing

Welcome to **Session 15** of the Applied Machine Learning Using Python curriculum!

As a Machine Learning Engineer, you can build models to predict *exactly* what a customer will do. But predicting what will happen is very different from understanding **why** it happens. 

If ice cream sales and shark attacks both increase in July, your predictive model will incorrectly assume that Ice Cream *causes* Shark Attacks. This session teaches you how to prove true causality using **A/B Testing** and advanced **Propensity Score Matching (PSM)**.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Define causal inference and explain the challenge of the Counterfactual.
2. Describe the fundamental difference between Randomized Controlled Trials (RCT) and Observational Studies.
3. Explain A/B testing and execute its step-by-step process in Python.
4. Explain Propensity Score Matching (PSM) and implement it using Logistic Regression.

---

## 📖 Part 1: Causal Inference & The Counterfactual Challenge

The fundamental problem of Causal Inference is the **Counterfactual**.
If you take a headache pill and your headache goes away, did the pill *cause* the cure? We will never truly know, because we cannot observe the "Counterfactual Universe" where you exactly replicated the timeline but *didn't* take the pill.

To prove causation, we must overcome **Confounding Variables**. (In the Ice Cream vs Shark Attack example, the confounding variable masking the truth is "Summer Weather/More people at the beach").

---

## 📖 Part 2: A/B Testing (The Gold Standard)

The most robust way to prove causal inference is a **Randomized Controlled Trial (RCT)**. In software and business, we call this an **A/B Test**.

Because users are split *randomly*, any confounding variables (like age, wealth, or geography) are equally distributed between Group A and Group B. Therefore, any difference in outcome must strictly be caused by the treatment!

### The Step-By-Step A/B Test Process:
1. **Hypothesis Formulation:** (e.g., "Changing the Buy button from Red to Green will increase conversions.")
2. **Randomization:** Splitting incoming traffic perfectly 50/50.
3. **Execution:** Running the test long enough to collect a large sample size.
4. **Measurement:** Calculating the Conversion Rates.
5. **Significance Analysis:** Using a statistical test (like a **T-Test**) to calculate the **p-value**. If `p < 0.05`, the result is statistically significant and not just random luck!

---

## 📖 Part 3: Propensity Score Matching (PSM)

What happens if you *cannot* randomly assign groups? 
Imagine trying to figure out if Smoking causes Cancer. You cannot ethically run an A/B test forcing half your users to smoke. You must rely on **Observational Data**, which is inherently biased.

**Propensity Score Matching (PSM)** solves this.
If you have a database of Smokers and Non-Smokers, they are usually drastically different demographics. PSM uses a **Logistic Regression** model to calculate a "Propensity Score" for every user (the mathematical probability that they *would* have smoked based on their age, diet, geography, etc). 

You then match every Smoker with a Non-Smoker who has the exact same Propensity Score. You have now artificially created a "Randomized" trial out of biased data!

---

## 🚀 Hands-On: Session Code Files

In the `code/` directory, you will find scripts demonstrating these Applied Python concepts:
1. **`01_ab_testing_ttest.py`**: Simulates an A/B test and uses `scipy` to automatically calculate the p-value.
2. **`02_propensity_score_matching.py`**: Uses Scikit-Learn to build a highly advanced PSM balancer on an observational dataset.

---
*© 2024 Aptech Limited — For Educational Use*
