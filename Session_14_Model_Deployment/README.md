# Session 14: Model Deployment and Maintenance

Welcome to **Session 14** of the Applied Machine Learning Using Python curriculum!

Up until now, you have been a Data Scientist working inside Jupyter Notebooks. But a model inside a notebook cannot be used by your company's customers! In this session, you transition into a **Machine Learning Engineer**.

You will learn how to extract an AI model out of a Python script, embed it inside a web server, and expose it via an API so that Mobile Apps, Web Frontends, and IoT devices can communicate with it remotely.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Explain model serialization and deserialization (`joblib`).
2. Describe Web APIs and their practices (REST, HTTP GET/POST).
3. Outline and build **Flask** applications for API development.
4. Explain Deployment architectures on Cloud Platforms (Docker containers).

---

## 📖 Part 1: Model Serialization & Deserialization

When you call `model.fit()`, the AI algorithm learns relationships in the data. However, the exact moment you close your python script, RAM clears, and that "learned intelligence" is permanently deleted!

**Serialization** is the process of converting that trained object in RAM into a byte stream and saving it permanently to your hard drive (usually as a `.pkl` or `.joblib` file). 
**Deserialization** is the exact reverse: loading that file back into a new Python script to make instant predictions without retraining.

*Note: For standard ML models, `joblib` is vastly superior to `pickle` because `joblib` is heavily optimized for large NumPy matrices.*

---

## 📖 Part 2: Web APIs (REST and JSON)

An **API** (Application Programming Interface) is essentially a digital waiter. 
- You (the client) give an order (a request).
- The waiter takes it to the Kitchen (the server/model).
- The Kitchen cooks the food (runs the prediction).
- The waiter brings it back to you (the response).

In modern architectures, we use **REST APIs**. They communicate over standard HTTP internet protocols.
- **GET Request:** Asking the API for information (e.g., "Give me the stock prices").
- **POST Request:** Sending data safely to the API to be processed (e.g., "Here is a securely encrypted JSON payload containing customer details; please predict if they will default on their loan").

The standard format for moving this data is **JSON** (JavaScript Object Notation), which natively translates to Python Dictionaries.

---

## 📖 Part 3: Flask for API Development

**Flask** is a micro web framework written in Python. It is unbelievably lightweight, meaning you don't need millions of files to launch a web server. 

With Flask, you can turn a standard python function into an internet endpoint using decorators (`@app.route`). When we combine Flask with a deserialized `joblib` model, we instantly create an AI microservice!

---

## 📖 Part 4: Cloud Deployment Architectures

You've built your Flask App. Now how do you put it on the internet?

Instead of paying for a dedicated virtual machine on AWS or Azure and manually configuring operating systems, modern Data Scientists use **Containers** and serverless hosting.

1. **Docker:** A technology that takes your model, your Flask app, and your exact Python environment (using `requirements.txt`) and packages it into an isolated "Container". This guarantees that your code will run exactly the same way on any server on the planet.
2. **Hugging Face Spaces:** A modern platform designed specifically for ML models. You can upload your Docker Container, and Hugging Face provides the hardware (and free GPUs!) to run your API live to the world.

---

## 🚀 Hands-On: Session Code Files

To properly run this session, you must execute the files in order!

1. Run **`01_train_and_serialize.py`**: This trains the model and saves it to a `.joblib` file.
2. Open **Terminal A** and run **`02_flask_api_server.py`**: This starts the web server. *Leave it running!*
3. Open **Terminal B** and run **`03_api_client_test.py`**: This script sends fake customer data across your local network into the Flask server and gets a live AI prediction back.

---
*© 2024 Aptech Limited — For Educational Use*
