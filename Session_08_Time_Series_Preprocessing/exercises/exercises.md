# Exercises: Time Series & Pre-processing

Test your conceptual understanding of sequential data modeling.

---

## 🟢 BEGINNER: Core Concepts

**Question 1:**
Why is it a mistake to use a standard Train/Test split (e.g., selecting 20% of your data completely at random) when working with Time Series data?

**Question 2:**
Identify whether the following represents a **Trend**, **Seasonality**, or **Noise**:
1. Ice cream sales peaking every July.
2. A company's overall revenue growing by 5% every year over a 10-year period.
3. A sudden dip in sales because a delivery truck broke down on a random Tuesday.

---

## 🟡 INTERMEDIATE: Model Selection

**Question 3:**
You work for a retail company that experiences massive, unpredictable sales spikes specifically around Thanksgiving and Black Friday. You also have a few missing weeks of data from when the server went down. 
Should you use an ARIMA model or Facebook's Prophet model? Justify your answer.

**Question 4:**
What does the "I" (Integration) in ARIMA stand for mathematically? Why do we difference our data before feeding it to the auto-regressive parts of the model? *(Hint: It's about stationarity).*

---

## 🔴 ADVANCED: Deep Learning (LSTM)

**Question 5:**
Explain the specific purpose of the **"Forget Gate"** inside an LSTM cell. What would happen to the neural network if the Forget Gate was completely broken and could never delete information?

**Question 6:**
When preparing data for an LSTM in Keras, the input shape must be 3-dimensional: `(samples, time_steps, features)`. Explain what `time_steps` means in this context in plain English.
