# 📊 Portfolio Component — Session 14

## Assignment: ML Pipeline Deployment (End-to-End)

### Overview
This is **Portfolio Project #5**. You have built exploratory notebooks, evaluation models, and dashboards. Now, you must build a true **Backend ML Microservice**.

You will train a model, wrap it in a pure `Flask` API, package everything perfectly using `Docker`, and deploy it live to the internet so that anyone in the world can send JSON requests to it.

---

### Deliverables

| # | Deliverable | Format | Estimated Time |
|---|------------|--------|----------------|
| 1 | Trained Model | `.joblib` | 30 mins |
| 2 | Flask API Server | `app.py` | 1.5 hours |
| 3 | Container Configuration | `Dockerfile` | 30 mins |
| 4 | Live Deployment | Hugging Face Docker Space | 30 mins |

---

### Project Requirements

#### 1. Choose a Dataset and Target
You can use the Iris dataset, the Titanic dataset, or the Predictive Maintenance dataset from Session 11. Train a model natively on your PC, and export it to `model.joblib`.

#### 2. Build the API (`app.py`)
Your Flask app must have two routes:
- `GET /`: Returns a simple welcome message (e.g., "Welcome to the ML Microservice API!").
- `POST /predict`: Accepts a JSON payload, converts it to a Pandas DataFrame, runs inference using the loaded model, and returns a JSON response containing the prediction.

#### 3. Containerize using Docker (`Dockerfile`)
Hugging Face Spaces supports raw Docker environments. In your project directory, you must create a file literally named `Dockerfile` (no extension).

**Dockerfile Template:**
```dockerfile
# 1. Use the official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your app (app.py, model.joblib)
COPY . .

# 5. Expose port 7860 (Hugging Face requires this specific port)
EXPOSE 7860

# 6. Run the Flask Webserver
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
```

#### 4. Deployment Instructions
1. Create a new Space on Hugging Face.
2. Select **Docker** (Blank) as the Space SDK.
3. Upload your `app.py`, `model.joblib`, `requirements.txt`, and `Dockerfile`.
4. Hugging Face will automatically read the Dockerfile, build an isolated cloud computer for you, and expose your Flask Web API to the world!

> 💡 **Why This Matters**: If you apply for a job as a Machine Learning Engineer and you can say *"I don't just train models in Jupyter notebooks; I deploy them behind scalable Dockerized Flask APIs"*, your resume moves to the top of the pile instantly.
