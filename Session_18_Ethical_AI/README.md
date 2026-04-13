# Session 18: Ethical Considerations in AI

Welcome to **Session 18** of the Applied Machine Learning Using Python curriculum!

As Data Scientists, our goal is to build algorithms that mathematically find patterns in data. But what happens if the data we feed the algorithm is historically biased? The algorithm will mathematically learn to be biased.

If you deploy a biased AI model to process Mortgages or Screen Resumes, your company will face significant legal consequences. In this session, you will learn to mathematically Audit your models for Ethics.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Define Bias in AI and its core types.
2. Identify how bias infiltrates an active AI practice.
3. Understand and calculate Responsible AI practices (Disparate Impact).

---

## 📖 Part 1: Defining Bias in AI

Bias occurs when an ML model systematically prejudices some groups over others.

### 1. Historical Bias
The data accurately reflects the world, but the world was biased.
**Example:** Amazon built an AI to screen resumes based on 10 years of past hiring data. Over the past 10 years, tech hiring was massively male-dominated. The AI mathematically learned that "being male" was heavily correlated with "being hired", and started penalizing resumes containing the word "Women's" (e.g. "Women's Chess Club Captain").

### 2. Representation Bias
The model performs worse for certain demographics because they were not sufficiently included in the training data.
**Example:** A facial recognition security system trained 80% on Caucasian faces and 20% on minority faces will statistically fail to recognize minority faces at a much higher rate.

### 3. Measurement (Proxy) Bias
You purposefully remove a protected class (like Race or Gender) from the dataset. However, another feature perfectly correlates with it.
**Example:** Redlining. An AI predicting loan defaults does not have access to 'Race', but it has access to 'Zip Code'. If cities are heavily segregated, the AI will use 'Zip Code' as a Proxy for Race, resulting in the exact same racist outcomes despite 'Race' explicitly being deleted from the data.

---

## 📖 Part 2: Responsible AI Practices

"Simply deleting the demographic column from the dataset does not make the AI fair." 

To practice Responsible AI, we must run automated **Audits** on our models using Fairness Metrics.

### The "Disparate Impact" Ratio (The Industry Metric)
This is the U.S. legal standard for analyzing fairness in employment and housing algorithms.
It compares the Approval Rate of the **Unprivileged Group** (e.g., minorities) against the Approval Rate of the **Privileged Group**.

*The Math:*
```
Disparate Impact = (Approval Rate of Minority) / (Approval Rate of Majority)
```

**The 80% Rule (Four-Fifths Rule):**
If the Disparate Impact ratio falls below `0.80`, the AI is legally considered Biased and must be taken offline and retrained.

---

## 🚀 Hands-On: Session Code Files

In the `code/` directory, you will find scripts demonstrating these Applied Python concepts:
1. **`01_detecting_bias.py`**: Trains a highly accurate Random Forest model on a mock Resume dataset and calculates its Disparate Impact ratio, proving that "High Predictive Accuracy" often hides severe ethical discrimination.
2. **`02_mitigating_bias.py`**: Fixes the biased model from Script 1 by identifying and dropping the "Proxy Variable" allowing the AI to pass the 80% Rule.

---
*© 2024 Aptech Limited — For Educational Use*
