# Capstone Portfolio Projects: End-to-End AI Engineering

Welcome to the final capstone module of the Applied Machine Learning Using Python course! 

To graduate and demonstrate your readiness for the industry, you must complete **one** of the following three end-to-end AI projects. You will act as a full-stack AI Engineer: dealing with data, training an unsupervised learning model, building a backend API to serve it, creating an interactive frontend for users, and deploying the entire system to public cloud platforms.

Choose **ONE** project from the list below.

---

## Project 1: Retail Customer Segmentation Engine
**Focus:** K-Means Clustering, Customer Behavior Profiling

**The Scenario:** 
A major e-commerce brand wants to move away from generic email campaigns. They have tasked you with building an AI engine that automatically segments customers based on their behavior (Annual Income, Age, Spending Score) so marketing can send personalized offers.

### Step-by-Step Instructions:

#### 1. Build and Train the AI Model
*   **Dataset:** Use a Customer Segmentation dataset (e.g., Mall Customers dataset).
*   **Preprocessing:** Scale your features using `StandardScaler`. This is critical for distance-based algorithms!
*   **Modeling:** Train a **K-Means Clustering** model. Use the Elbow Method and Silhouette Score to determine the optimal number of clusters (`K`).
*   **Profiling:** Write a script to interpret the clusters (e.g., Cluster 0 = "High Income, Low Spender").
*   **Serialization:** Use the `joblib` library to save both your trained `KMeans` model and your `StandardScaler` to disk (as `.pkl` files).

#### 2. Build the Python Backend (API)
*   **Framework:** Initialize a robust **Flask** application.
*   **Loading:** When your Flask app starts, load the `.pkl` files using `joblib`.
*   **Endpoint:** Create a POST endpoint `/api/segment`. It should:
    *   Accept a JSON payload: `{"age": 30, "income": 60000, "spending_score": 85}`.
    *   Convert this into a NumPy array or DataFrame.
    *   Transform the input using the loaded `StandardScaler`.
    *   Predict the cluster using the `KMeans` model.
    *   Return a JSON response with the cluster ID and a human-readable segment name.

#### 3. Build the Interactive Frontend
*   **Technology:** HTML5, modern CSS, and plain JavaScript (Fetch API).
*   **Design:** Create a vibrant, clean marketing-dashboard interface. Use a high-quality color palette (e.g., deeper purples and gradients) and modern typography (like Google 'Inter' font).
*   **Functionality:** Build a form with sliders or clean input fields for Age, Income, and Spending Score. When the user clicks "Analyze Customer", use JavaScript's `fetch()` to call your Flask API and dynamically render an engaging "Result Card" with micro-animations.

#### 4. Deploy to the Web
*   **Backend Deployment:** Push your Flask API code to GitHub. Deploy it as a Web Service on **Render** or **Railway**. *Ensure you update your frontend's `fetch()` URL to point to this live API.*
*   **Frontend Deployment:** Deploy your static frontend code (HTML/CSS/JS) to **Netlify** or **Vercel**.
*   **Deliverable:** Submit your public frontend live link and your GitHub repository.

---

## Project 2: Real-time Financial Anomaly Detector
**Focus:** DBSCAN / Isolation Forest, Fraud & Outlier Detection

**The Scenario:**
A fintech startup is losing money to fraudulent transactions, but they don't have historical "fraud" labels. You are tasked with building a real-time unsupervised anomaly detection system that relies on transactional density to flag highly unusual operations as soon as they happen.

### Step-by-Step Instructions:

#### 1. Build and Train the AI Model
*   **Dataset:** Use a mock financial transaction dataset with features like `Transaction_Amount`, `Distance_From_Home`, and `Time_Since_Last_Transaction`.
*   **Preprocessing:** Apply `RobustScaler` to ensure extreme anomalies don't distort your scaling process.
*   **Modeling:** Train a **DBSCAN** (or Isolation Forest) model. Tune the `eps` and `min_samples` parameters to successfully separate dense "normal" transaction zones from isolated "noise" points (-1).
*   **Serialization:** Since DBSCAN doesn't naturally predict on *new* data out of the box in `sklearn`, you must serialize a trained `IsolationForest` or a custom logic pipeline using `NearestNeighbors` along with your `RobustScaler` via `joblib`.

#### 2. Build the Python Backend (API)
*   **Framework:** Set up a **Flask** API.
*   **Loading:** Load the serialized model and scaler on startup.
*   **Endpoint:** Create a POST endpoint `/api/verify_transaction`. It should:
    *   Accept JSON: `{"amount": 5000, "distance": 120, "time_delta": 2}`.
    *   Scale the live data.
    *   Predict if it is an anomaly (e.g., returns -1).
    *   Return JSON: `{"is_fraudulent": true, "risk_level": "CRITICAL"}`.

#### 3. Build the Interactive Frontend
*   **Technology:** HTML5, CSS (Dark Mode), and vanilla JavaScript.
*   **Design:** Build a sleek, "Hacker/Security Operations Center" dark-mode theme. Use neon green for clean transactions and blinking crimson red and warning icons for anomalies.
*   **Functionality:** Create an input terminal interface. When submitting the transaction, show a "Scanning..." loading state. Use `fetch()` to call the API. If the API returns `is_fraudulent: true`, trigger a high-alert red interface layout change.

#### 4. Deploy to the Web
*   **Backend Deployment:** Deploy your Flask app to **Render** or **Railway**, ensuring you configure CORS properly so your frontend can communicate with it.
*   **Frontend Deployment:** Deploy your static dashboard to **Netlify** or **Vercel**.
*   **Deliverable:** Submit your public-facing URL where anyone can mock a fraud attempt and see the system react.

---

## Project 3: User Engagement & Churn Risk Profiling
**Focus:** Gaussian Mixture Models (GMM), Soft Clustering

**The Scenario:**
A SaaS company wants to understand user engagement. Customers aren't just "active" or "inactive"—there is a spectrum of behaviors. Using GMM, build a system that assigns probabilities of a user belonging to different engagement tiers, instantly flagging users with a high probability of churning.

### Step-by-Step Instructions:

#### 1. Build and Train the AI Model
*   **Dataset:** Use a user engagement dataset featuring metrics like `Logins_Per_Week`, `Feature_Usage_Time`, and `Support_Tickets_Opened`.
*   **Preprocessing:** Standardize the features.
*   **Modeling:** Train a **Gaussian Mixture Model (GMM)** to accomplish "soft clustering," defining components like "Power Users," "Standard Users," and "At-Risk/Churning Users."
*   **Serialization:** Export your trained `GaussianMixture` object and your scaler using `joblib`.

#### 2. Build the Python Backend (API)
*   **Framework:** Develop a **Flask** web server.
*   **Loading:** Initialize your pre-trained models into memory when the script runs.
*   **Endpoint:** Expose a POST endpoint `/api/user_profile`.
    *   Accept JSON data containing a user's recent engagement metrics.
    *   Use the GMM's `predict_proba()` function to get the percentage breakdown (e.g., 80% chance they are "At-Risk", 20% "Standard").
    *   Return the full probability distribution mapping in your JSON response.

#### 3. Build the Interactive Frontend
*   **Technology:** HTML5, modern CSS, JavaScript (Fetch API + Chart.js).
*   **Design:** Create a premium "Customer Success Dashboard" with a glassmorphism aesthetic. Use clean whites, subtle shadows, and a polished, professional look. 
*   **Functionality:** Include an input module for the user's weekly stats. Upon submission, invoke the backend API. Take the returned probability percentages and dynamically render a **Chart.js Donut Chart** showing the user's risk profile (e.g., a pie chart showing 80% At-Risk in red, 20% Standard in yellow).

#### 4. Deploy to the Web
*   **Backend Deployment:** Host your Python app on **Render** or **Railway**. 
*   **Frontend Deployment:** Push your code to **Vercel** or **Netlify**.
*   **Deliverable:** Submit your live link demonstrating the beautiful, probabilistic customer profiling system.

---

> **Note on Deployment Guidelines:**
> - Ensure your `requirements.txt` includes `flask`, `scikit-learn`, `flask-cors`, `pandas`, `numpy`, and `joblib`.
> - Always test your API locally using tools like Postman before attempting to deploy.
> - Remember to handle CORS (Cross-Origin Resource Sharing) in your Flask API by using `flask_cors`, or your frontend will be blocked from accessing it!
