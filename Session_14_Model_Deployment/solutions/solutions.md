# Session 14 — Solutions

## 🟢 Solution 14.1/14.2: Joblib vs Pickle

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import pickle

# Train
X = np.random.rand(10, 1)
y = X * 2
model = LinearRegression().fit(X, y)

# EX 14.1: Joblib (Industry Standard for Scikit-Learn)
joblib.dump(model, 'my_regressor.joblib')

# EX 14.2: Pickle (Standard Python)
with open('my_regressor.pkl', 'wb') as file:
    pickle.dump(model, file)
```

---

## 🟡 Solution 14.3: Basic Routing (Flask)

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "pong!"})

if __name__ == "__main__":
    app.run(port=5001)
```

---

## 🟡 Solution 14.4: POST Requests

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/echo', methods=['POST'])
def echo():
    data = request.get_json()
    
    # Add our custom key
    data['server_message'] = "Data received perfectly!"
    
    return jsonify(data)

if __name__ == "__main__":
    app.run(port=5001)
```

---

## 🔴 Solution 14.5: The End-To-End API

```python
# --- SERVER SCRIPT ---
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
# Assuming 'iris_model.joblib' was created earlier
model = joblib.load('iris_model.joblib') 

@app.route('/predict_iris', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame([data])
        
        prediction = model.predict(df)[0]
        
        classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        
        return jsonify({
            "status": "success",
            "prediction": classes[prediction]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()
```
