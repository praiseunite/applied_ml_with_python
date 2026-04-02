# Session 07: Handling Imbalanced Data & Dimensionality Reduction

Welcome to **Session 07** of the Applied Machine Learning Using Python curriculum!

In a perfect world, a dataset classifying "Cats" and "Dogs" will have exactly 500 pictures of cats and 500 pictures of dogs. The real world doesn't work like this. Real-world datasets—especially in cybersecurity, medicine, and finance—are heavily skewed. For instance, out of 100,000 credit card transactions, only 10 might be fraudulent!

In this session, we will learn how to handle this imbalance, and how to condense extremely wide datasets using Feature Selection and PCA.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Explain different strategies for feature selection and common approaches.
2. Define PCA (Principal Component Analysis) and its importance.
3. Describe and implement oversampling and under-sampling techniques.

---

## 🟢 BEGINNER: Feature Selection Strategies

If your dataset has 1,000 columns (Features), feeding all of them into a Machine Learning model is a bad idea. It slows down training and confuses the model mathematically (the Curse of Dimensionality). 

We fix this via **Feature Selection**. There are three main approaches:

1. **Filter Methods:** Fast and simple. We use statistical tests (like Correlation or Chi-Square) to see if a feature actually impacts the target variable. If a column has no correlation, we delete it before training even starts.
2. **Wrapper Methods:** We train the model over and over again, systematically adding or removing features (like *Recursive Feature Elimination*) to see which exact combination gives the highest accuracy. It is highly accurate but computationally expensive.
3. **Embedded Methods:** The model selects the features *while* it is training! For example, algorithms like LASSO Regression or Random Forests automatically assign "Feature Importance" scores and ignore useless data natively.

---

## 🟡 INTERMEDIATE: PCA (Principal Component Analysis)

What if you have 100 features, but they are all slightly useful, and you cannot afford to just delete them?

**Definition:** PCA is an Unsupervised Dimensionality Reduction technique. Instead of deleting features, it mathematical *compresses* them.
**The Analogy:** Imagine holding a 3-Dimensional coffee mug. If you shine a bright flashlight at it, it casts a 2-Dimensional shadow on the wall. The shadow isn't the mug, but it preserves the *shape* and *variance* of the mug almost perfectly! 

**Importance:**
- PCA takes a 100-column dataset and shines a mathematical flashlight on it, projecting it into a 5-column dataset.
- This allows models to train incredibly fast.
- It allows us to visualize complex 50-dimensional data on a flat 2D graph!

---

## 🔴 ADVANCED: Imbalanced Data (Over/Under Sampling)

Let's return to the Credit Card Fraud problem (99,990 normal transactions vs 10 frauds).
If your Machine Learning model literally just hardcodes the answer "Everything is Normal," it achieves **99.99% Accuracy**! But it is a terrible model because it completely failed its job. 

We must balance the dataset *before* training.

1. **Under-sampling:** We randomly delete normal transactions until we only have 10 normal transactions left, matching the 10 fraud transactions. 
   - *Pros:* Fast. Data is 50/50.
   - *Cons:* We just threw away 99,980 pieces of valuable data! The model will be completely starved of information.
2. **Over-sampling (SMOTE):** Synthetic Minority Over-sampling Technique. Instead of deleting data, we use an algorithm to synthesize fake context. It analyzes the 10 fraud transactions, figures out their mathematical patterns, and artificially hallucinates 99,980 brand-new, realistic fraud transactions!
   - *Pros:* No data loss. The model learns incredibly well.
   - *Cons:* Prone to overfitting if done incorrectly.

---

## 💻 HANDS-ON LAB: Balancing the Scales

Open the Jupyter Notebook provided in this session: `notebooks/01_Imbalanced_Data_Lab.ipynb`.

**What you will do:**
1. Generate an extremely imbalanced dataset (e.g., 98% Class A, 2% Class B).
2. Train an algorithm and prove that standard accuracy fails drastically.
3. Use the `imbalanced-learn` library to perform SMOTE over-sampling.
4. Use `scikit-learn` to perform PCA, compressing a synthetic dataset and plotting it in 2D space.

---

## 📚 FURTHER READING
- **SMOTE Original Paper:** "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)
- **PCA Visualization:** Scikit-Learn Documentation on Manifold Learning.

---
*© 2024 Aptech Limited — For Educational Use*
