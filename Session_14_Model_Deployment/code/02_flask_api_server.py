from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize Flask App
app = Flask(__name__)

# Global variable to hold the model
MODEL = None
MODEL_PATH = 'my_production_model.joblib'

# 1. App Initialization & Deserialization
def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        # Deserialization! Loading the bytes back into RAM
        MODEL = joblib.load(MODEL_PATH)
        print(f"✅ Loaded previously serialized model from {MODEL_PATH}")
    else:
        print(f"❌ CRITICAL ERROR: Could not find {MODEL_PATH}.")
        print("Please run `01_train_and_serialize.py` first!")
        exit(1)

# 2. Define a Health Check endpoint (HTTP GET)
@app.route('/health', methods=['GET'])
def health_check():
    """Simple endpoint to verify if the server is awake."""
    return jsonify({
        "status": "healthy",
        "api_version": "1.0",
        "model_loaded": MODEL is not None
    })

# 3. Define the Prediction endpoint (HTTP POST)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload. Returns the AI's prediction.
    """
    try:
        # A. Get the raw JSON data securely sent via HTTP POST
        json_data = request.get_json()
        
        # B. Convert JSON back into a format the model understands (Pandas)
        # Note: In real life, you must validate that the expected columns actually exist!
        df = pd.DataFrame([json_data])
        
        # C. Make the prediction using the Deserialized model
        prediction_array = MODEL.predict(df)
        prediction_prob = MODEL.predict_proba(df)[0]
        
        # Determine confidence of the predicted class
        predicted_class = int(prediction_array[0])
        confidence = float(prediction_prob[predicted_class])
        
        # D. Return the result back over the internet as JSON
        result = {
            "prediction": predicted_class,
            "confidence": round(confidence, 3),
            "status": "success"
        }
        return jsonify(result), 200 # 200 is the HTTP code for OK

    except Exception as e:
        # Error handling is critical for APIs so they don't crash the server
        error_response = {
            "status": "error",
            "message": str(e)
        }
        return jsonify(error_response), 400 # 400 is HTTP code for Bad Request

if __name__ == "__main__":
    print("="*60)
    print(" Starting Flask API Server... ".center(60))
    print("="*60)
    
    # Load model into RAM before starting the server
    load_model()
    
    print("\n🚀 Server is running on port 5000!")
    print("⚠️  DO NOT CLOSE THIS TERMINAL.")
    print("👉 Open a NEW terminal to run `03_api_client_test.py`\n")
    
    # Start the webserver
    # In production, do not use app.run() - use wsgiref or gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
