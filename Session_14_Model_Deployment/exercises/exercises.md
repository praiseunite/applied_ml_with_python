# Session 14 — Exercises

## 🟢 Beginner Exercises

### Exercise 14.1: The Joblib Dump
**Objective**: Practice standard serialization.

**Instructions**:
Write a python script that trains a very simple `LinearRegression` model using `sklearn.linear_model`. 
Save the model specifically using standard `joblib` into a file named `my_regressor.joblib`.

### Exercise 14.2: Standard Built-in Pickle
**Objective**: Understand the difference between joblib and standard python.

**Instructions**:
Write a script that loads the `my_regressor.joblib` file you created in Exercise 14.1.
Now, re-save it, but this time use the standard python `pickle` library. Save it to `my_regressor.pkl`.

---

## 🟡 Intermediate Exercises

### Exercise 14.3: Basic Routing (Flask)
**Objective**: Write your first API server.

**Instructions**:
Create a file called `basic_server.py`.
1. Import `Flask`.
2. Create an `@app.route('/ping', methods=['GET'])`.
3. Make the function return a simple JSON response: `{"status": "pong!"}` using `jsonify`.
4. Run the app on port `5001`.

### Exercise 14.4: POST Requests
**Objective**: Accept data from the internet safely.

**Instructions**:
In your `basic_server.py`, add a secondary route called `@app.route('/echo', methods=['POST'])`.
1. Use `request.get_json()` to capture the data.
2. Return exactly the data they sent you, but add a new key `"server_message": "Data received perfectly!"`.
3. Try pinging it with `requests` in a different script to prove it works.

---

## 🔴 Advanced Exercises

### Exercise 14.5: The End-To-End API
**Objective**: Build a real ML Microservice.

**Instructions**:
1. Train a model on the `iris` dataset and save it.
2. Build a Flask app with a `/predict_iris` route.
3. The POST request will send JSON like: `{"sepal_length": 5, "sepal_width": 3, "petal_length": 1, "petal_width": 0.2}`.
4. Your API must load the JSON, convert to a dataframe, predict the class name (Setosa, Versicolor, Virginica), and return it over HTTP.

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`).
- Run your server on `port=5000` or `port=5001`.
- Name your files `exercise_14_X.py`.
