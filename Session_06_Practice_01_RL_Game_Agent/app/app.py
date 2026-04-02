import streamlit as st
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import os

st.set_page_config(page_title="RL Game Agent Dashboard", page_icon="🎮")

st.title("🎮 RL Game Agent: CartPole")
st.markdown("""
This application demonstrates a **Reinforcement Learning** agent trained to balance a pole on a moving cart.
* **Blue Box:** The Agent Brain (PPO Model)
* **Environment:** CartPole-v1
""")

# Check if the model exists (students must train it first!)
MODEL_PATH = "ppo_cartpole"
MODEL_FILE = f"{MODEL_PATH}.zip"

if not os.path.exists(MODEL_FILE):
    st.error(f"⚠️ Model not found! Please run the training notebook in the `notebooks/` folder first, ensuring it saves `{MODEL_FILE}` into this `app/` folder.")
    st.stop()

st.success("✅ Trained Agent Brain loaded successfully!")

if st.button("▶️ Run Agent Simulation", type="primary"):
    # Create the environment with RGB rendering so Streamlit can capture frames
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # Load the trained model
    model = PPO.load(MODEL_PATH)
    
    obs, info = env.reset()
    
    # Placeholder in the UI for the video frames
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    score = 0
    max_steps = 500
    
    for i in range(max_steps):
        # 1. Ask the Brain what to do
        action, _states = model.predict(obs, deterministic=True)
        
        # 2. Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        # 3. Capture what the environment looks like now
        frame = env.render()
        
        # 4. Display the frame in Streamlit
        frame_placeholder.image(frame, channels="RGB", caption=f"Timestep: {i} | Score: {score}")
        status_text.text(f"Balancing... current score: {score}")
        
        # Slow down slightly so we can watch it
        time.sleep(0.02)
        
        if terminated or truncated:
            break
            
    env.close()
    
    if score >= 490:
        st.balloons()
        st.success(f"🏆 Amazing! The agent balanced the pole like a pro. Final Score: {score}")
    else:
        st.info(f"Simulation ended. Final Score: {score}. (Perfect score is 500)")
