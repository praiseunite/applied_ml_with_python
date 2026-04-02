# Solutions: Time Series & Pre-processing

---

## 🟢 BEGINNER: Core Concepts

**Answer 1:**
Because time series relies on the chronological sequence of events. If you pull out 20% of the data randomly, you are looking into the future to predict the past, causing "data leakage." You must split time series chronologically (e.g., train on 2018-2022, test on 2023).

**Answer 2:**
1. Ice cream sales peaking every July = **Seasonality** (A predictable, repeating short-term cycle).
2. Revenue growing 5% every year = **Trend** (The long-term overarching direction).
3. Delivery truck breaking down = **Noise / Residual** (A spontaneous, unpredictable event).

---

## 🟡 INTERMEDIATE: Model Selection

**Answer 3:**
You should absolutely use **Facebook's Prophet**. 
ARIMA models will likely crash or fail if there are missing weeks of data, and they have no native way to understand specific holiday dates like Black Friday. Prophet was specifically designed to handle missing data gracefully and allows you to explicitly input a list of holidays so it expects the massive sales spikes instead of treating them as noise.

**Answer 4:**
Integration ("I") refers to *Differencing* the data (subtracting yesterday's value from today's value). Time series models require data to be "Stationary" (meaning the mean and variance don't constantly float upwards over time). Differencing removes the long-term trend, flattening the data so the Auto-Regressive components can accurately analyze the remaining patterns.

---

## 🔴 ADVANCED: Deep Learning (LSTM)

**Answer 5:**
The Forget Gate's job is to look at the long-term memory of the cell and explicitly decide what information is outdated or no longer needed. If it was broken, the memory cell would just continuously accumulate every single piece of data from the beginning of time. Eventually, the signal would be so overloaded with irrelevant historical data that the network would become mathematically paralyzed and unable to make accurate modern predictions.

**Answer 6:**
`time_steps` is your "Lookback Window." In plain English, it answers the question: *"How many days into the past should the model look to predict the very next day?"* For example, if `time_steps = 10`, the model will analyze January 1st through January 10th to generate the prediction for January 11th.
