import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb # Required to ensure the XGBoost loader works

print("Starting Industrial IoT Dashboard...")

# 1. Load the Saved Model
MODEL_PATH = "saved_model.joblib"

# --- SAFETY NET FOR DEPLOYMENT ---
# If a student runs the app without doing the notebook first, we train a rapid dummy model 
# so the UI still functions perfectly and doesn't crash on Hugging Face Spaces.
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ {MODEL_PATH} not found. Training a quick dummy XGBoost model for deployment demonstration...")
    # Generate dummy data
    X_dummy = pd.DataFrame(np.random.rand(100, 7), columns=[
        'Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
        'Temp_Difference', 'Power_Proxy'
    ])
    y_dummy = np.random.randint(0, 2, 100)
    dummy_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    dummy_model.fit(X_dummy, y_dummy)
    joblib.dump(dummy_model, MODEL_PATH)
    print("✅ Dummy model saved.")
# ----------------------------------

model = joblib.load(MODEL_PATH)
print("✅ Primary Model Loaded Successfully.")

def predict_failure(machine_type, air_temp, proc_temp, rot_speed, torque, tool_wear):
    """
    Takes live inputs from the Gradio UI, engineers the necessary features exactly 
    like the training notebook, and returns a prediction.
    """
    try:
        # Encode Machine Type: L=0, M=1, H=2
        type_encoded = {'Low (L)': 0, 'Medium (M)': 1, 'High (H)': 2}.get(machine_type, 0)
        
        # Calculate Engineered Features
        temp_diff = proc_temp - air_temp
        power_proxy = rot_speed * torque
        
        # Assemble DataFrame matching training features
        input_data = pd.DataFrame({
            'Type': [type_encoded],
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [proc_temp],
            'Rotational speed [rpm]': [rot_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear],
            'Temp_Difference': [temp_diff],
            'Power_Proxy': [power_proxy]
        })
        
        # Predict probability of failure
        prob_failure = model.predict_proba(input_data)[0][1]
        
        if prob_failure > 0.60: # Threshold
            return f"🚨 CRITICAL: Failure Imminent! ({prob_failure*100:.1f}% confidence). Immediate maintenance required."
        elif prob_failure > 0.35:
            return f"⚠️ WARNING: High Stress detected ({prob_failure*100:.1f}% probability). Schedule checks soon."
        else:
            return f"✅ NORMAL: Operating within safe parameters. ({prob_failure*100:.1f}% failure probability)."
            
    except Exception as e:
        return f"Error computing prediction: {str(e)}"

# 2. Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏭 Industrial IoT Predictive Maintenance Dashboard")
    gr.Markdown("Real-time monitoring using XGBoost Ensembles. Adjust the sensor sliders to see how the Machine Learning model responds to operational stress.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📡 Live Sensor Data Feed")
            machine_type = gr.Radio(["Low (L)", "Medium (M)", "High (H)"], value="Low (L)", label="Machine Quality Class")
            air_temp = gr.Slider(290.0, 310.0, value=298.1, label="Air Temperature [K]")
            proc_temp = gr.Slider(300.0, 320.0, value=308.6, label="Process Temperature [K]")
            rot_speed = gr.Slider(1000, 3000, value=1400, label="Rotational Speed [rpm]")
            torque = gr.Slider(10.0, 80.0, value=40.0, label="Torque [Nm]")
            tool_wear = gr.Slider(0, 300, value=50, label="Tool Wear [minutes]")
            
            predict_btn = gr.Button("🔍 Run AI Diagnostics", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 🤖 System Evaluation Status")
            output_status = gr.Textbox(label="Status Report", lines=3, scale=2)
            
            # Example values that trigger a failure in the dataset
            gr.Markdown(
                "**💡 Try failing the machine:**<br>"
                "- Max out the Tool Wear to `250+` (Fatigue).<br>"
                "- Max out Rotational Speed and Torque simultaneously (Power Failure).<br>"
                "- Set Process Temp far greater than Air Temp."
            )

    predict_btn.click(
        predict_failure, 
        inputs=[machine_type, air_temp, proc_temp, rot_speed, torque, tool_wear], 
        outputs=output_status
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
