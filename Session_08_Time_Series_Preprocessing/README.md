# Session 08: Time Series Data Pre-processing & Models

Welcome to **Session 08** of the Applied Machine Learning Using Python curriculum!

Most machine learning assumes that your data points have nothing to do with each other (e.g., patient #1's blood pressure doesn't affect patient #2's). **Time Series** breaks this rule entirely. In time series, data is collected sequentially over time, meaning *yesterday's data deeply influences today's data*.

In this session, we will explore how to predict the future based on the past using three progressive models: ARIMA, Prophet, and LSTMs.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Explain what makes Time Series data unique (Trends, Seasonality, and Noise).
2. Understand Facebook's **Prophet** model and its incredible business value.
3. Define **LSTM (Long Short-Term Memory)** neural networks using a simple "human memory" analogy.
4. Implement and evaluate time series models to forecast stock prices or weather patterns.

---

## 🟢 BEGINNER: The Intuition of Time Series

If you want to predict umbrella sales, you cannot just look at random factors. You must look at the time of year. Umbrella sales will naturally spike every spring and drop in the summer. 

This brings us to the three components of any time series:
1. **Trend:** The overall long-term direction (e.g., umbrella sales are growing 5% year-over-year).
2. **Seasonality:** Repeating short-term cycles (e.g., umbrella sales spike every April).
3. **Noise (Residuals):** Completely random events you cannot predict (e.g., a sudden freak hurricane).

**The Traditional Model: ARIMA**
ARIMA stands for Auto-Regressive Integrated Moving Average. It was the gold standard for decades. It beautifully models Trends and Seasonality by looking back at previous time steps. However, it physically breaks if there are missing days in your dataset, and it has no concept of "Holidays" (like a sudden spike in sales just because it's Christmas).

---

## 🟡 INTERMEDIATE: Facebook's Prophet Model

In 2017, Facebook (Meta) open-sourced a model called **Prophet**. It completely revolutionized business forecasting because it solves the exact problems ARIMA struggles with.

### Why is Prophet so Important?
1. **Missing Data is Fine:** In the real world, sensors break and stores close on Sundays. Prophet gracefully handles missing dates without crashing.
2. **Built-in Holidays:** Real-world sales are heavily driven by holidays (Black Friday, Christmas). Prophet allows you to easily inject country-specific holidays into the model so it knows exactly when to expect massive, unnatural spikes in the data.
3. **Interpretable:** You don't need a math degree to use it. It outputs beautiful, readable graphs showing exactly what the daily, weekly, and yearly trends are.

*Because of this, Prophet is the absolute weapon of choice for corporate data analysts.*

---

## 🔴 ADVANCED: LSTM (Memory in Neural Networks)

What if predicting the future requires understanding a highly complex, nonlinear pattern that happened 30 days ago? Standard algorithms fail here.

To solve this, we use **LSTM (Long Short-Term Memory)** neural networks. 

### The Amnesia Problem
Standard neural networks suffer from total "amnesia." When they look at Day 5, they have completely forgotten what happened on Day 1. If you are reading a book, you only understand chapter 5 because you *remember* chapters 1 through 4.

### The LSTM "Memory Cell" Solution
LSTMs fix neural amnesia by acting like a human brain. Inside the network, they use tiny bouncers called **Gates**:
- **The Forget Gate:** Looks at old memories and says, "This old trend is no longer relevant, delete it."
- **The Input Gate:** Looks at today's new data and says, "This is really important, store it in our long-term memory vault!"
- **The Output Gate:** Looks at the memory vault and decides what the prediction for tomorrow should be.

Because LSTMs selectively remember and forget data over long periods, they are the backbone of modern AI (used for everything from predicting the stock market to generating text in ChatGPT's predecessors).

---

## 💻 HANDS-ON LAB: Forecasting the Future

Open the Jupyter Notebook provided in this session: `notebooks/01_Time_Series_Lab.ipynb`.

**Important Note on Environments:**
Installing the C++ compilers required for `Prophet` on a local Windows machine can sometimes be tricky. If you encounter installation errors locally, simply upload your `01_Time_Series_Lab.ipynb` notebook to **Google Colab** where Prophet and TensorFlow are pre-installed!

**What you will do:**
1. Load historical Sunspot or Stock Market sequential data.
2. Fit Facebook's Prophet to visualize the exact yearly trends.
3. Prepare sequential windows (e.g., predicting Day 11 using Days 1-10) to feed into a Keras LSTM model.

---

## 📚 FURTHER READING
- **Prophet Documentation:** https://facebook.github.io/prophet/
- **Understanding LSTMs:** "Understanding LSTM Networks" (Christopher Olah's Blog - The industry standard visual explanation).

---
*© 2024 Aptech Limited — For Educational Use*
