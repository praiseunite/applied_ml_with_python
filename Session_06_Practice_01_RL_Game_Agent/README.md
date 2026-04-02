# Session 06: Practice 01 – Deploying an RL Game Agent

Welcome to **Session 06** of the Applied Machine Learning Using Python curriculum! 

In Sessions 1 through 5, you learned the fundamentals of Machine Learning, Advanced Clustering, and Reinforcement Learning (MDPs and Policy Optimization). Now, it is time to put theory into practice. This is a dedicated 2-hour **Try It Yourself** practical session where you will build the "Brain" of an RL Agent, wrap it in a software frontend, and deploy it to the cloud.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Train a Reinforcement Learning agent using Stable-Baselines3 (PPO algorithm).
2. Save and export a trained Neural Network as a persistent artifact.
3. Build a software frontend using Streamlit to load the "AI Brain" and observe its gameplay.
4. Deploy the full interactive application to Hugging Face Spaces.
5. Understand the critical architectural boundary between ML Model Training and ML Inference / Deployment.

---

## 📖 Part 1: Train the AI Model (The Brain)

Before deploying, our agent needs to learn how to play. We will teach an AI agent to perfectly balance a pole on a moving cart using **Reinforcement Learning**.

1. Navigate to the `notebooks/` directory.
2. Open the comprehensive Jupyter Notebook: `RL_Game_Agent_Lab.ipynb`.
3. Follow the guided cells to initialize the `CartPole-v1` environment.
4. Run the PPO algorithm for 25,000 timesteps.
5. **Artifact Generation:** The notebook will export the trained model as `ppo_cartpole.zip` directly into the `app/` directory. This is our self-contained "AI Brain".

---

## 📖 Part 2: Build a Local Frontend & Link the Model

We must switch from "Data Scientist" mode to "Software Engineer" mode to give our Brain a frontend.

1. Navigate to the `app/` directory.
2. Review the `app.py` script. It uses **Streamlit** to create a web interface.
3. Observe how the frontend code seamlessly loads the `ppo_cartpole.zip` file, connects to the game environment, and asks the AI for continuous predictions on what move to make next.
4. **Run the Simulation Locally:**
   ```bash
   cd app
   pip install -r requirements.txt
   streamlit run app.py
   ```
   *Verify that the app launches in your browser and the AI successfully balances the pole in real-time!*

---

## 📖 Part 3: Deploy to Hugging Face Spaces

Now that your web application works locally, it's time to share it globally!

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and log in to your account.
2. Create a new Space and select **Streamlit** as your Space SDK.
3. Upload the exact contents of your `app/` folder (including `app.py`, `requirements.txt`, and the `ppo_cartpole.zip` artifact model) directly into the Hugging Face repository.
4. Watch Hugging Face build the Docker container and deploy your app. 
5. Congratulations, you just published an AI Agent portfolio project!

---

## 🚀 Hands-On: Session Files

In this folder, you will find the complete workflow structure:

1. **`notebooks/RL_Game_Agent_Lab.ipynb`**
   The training environment where the mathematical optimization happens.
2. **`app/app.py`**
   The deployment script that wraps the trained model in a beautiful interactive UI.
3. **`app/requirements.txt`**
   The exact package definitions required for the web server to run your code successfully.

---

## 🛠️ Environment Prerequisites

Ensure you have activated your virtual environment before starting:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

The app execution requires `streamlit`, `stable-baselines3`, and `gymnasium`.

---
*© 2024 Aptech Limited — For Educational Use*
