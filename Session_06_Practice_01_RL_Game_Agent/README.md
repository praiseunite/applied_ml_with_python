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

## 🛠️ Step 0: Environment Setup (Start Here!)

Before doing anything, you need to set up and activate your Python environment. You must do this from the **Root Project Folder**, not from inside the Session 6 folder.

1. Open your terminal (PowerShell, Command Prompt, or standard terminal on Mac/Linux).
2. Navigate to the absolute root folder of the entire course:
   ```bash
   cd path/to/Applied_ML_with_python
   ```
3. **If you have never run the setup before**, build the environment by running the setup script:
   ```powershell
   # Windows
   .\setup.bat

   # macOS / Linux
   ./setup.sh
   ```
4. **Activate the virtual environment**:
   ```powershell
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1

   # Windows (Command Prompt / CMD)
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```
   *(You will know this step worked if your terminal prompt now starts with a green `(venv)` tag!)*

---

## 📖 Part 1: Train the AI Model (The Brain)

Now that your environment is activated, our agent needs to learn how to play. We will teach an AI agent to perfectly balance a pole on a moving cart using **Reinforcement Learning**.

1. In your same terminal, change your directory to the Session 06 folder:
   ```bash
   cd Session_06_Practice_01_RL_Game_Agent
   ```
2. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```
3. In the Jupyter Lab browser that opens, navigate to the `notebooks/` directory and open `RL_Game_Agent_Lab.ipynb`.
4. Read through the notebook and **run all the cells** from top to bottom. It will initialize the `CartPole-v1` environment and mathematically train the PPO algorithm.
5. **Artifact Generation:** The final notebook cell will export the trained model as `ppo_cartpole.zip` directly into the `app/` directory. This is our self-contained "AI Brain".

---

## 📖 Part 2: Build a Local Frontend & Link the Model

We must switch from "Data Scientist" mode to "Software Engineer" mode to give our Brain a frontend.

1. Close Jupyter Lab (by typing `Ctrl+C` in your terminal and typing `y`), or simply open a **new** terminal window and activate the virtual environment again (Step 0).
2. Make sure you are inside the `app/` folder for Session 6:
   ```bash
   cd app
   ```
   *(If you opened a new terminal at the project root, type: `cd Session_06_Practice_01_RL_Game_Agent/app`)*
3. Install the specific requirements for the web frontend:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```
5. **View the Simulation:** Streamlit will start a server and output a **Local URL** (usually `http://localhost:8501`). Open that URL in your web browser.
   *Verify that the app successfully launches and observe the AI balancing the pole based on your newly trained model!*

---

## 📖 Part 3: Deploy to Hugging Face Spaces

Now that your web application works locally, it's time to share it globally!

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and log in to your account.
2. Create a new Space and select **Streamlit** as your Space SDK.
3. Upload the exact contents of your local `app/` folder (including `app.py`, `requirements.txt`, and the `ppo_cartpole.zip` artifact model) directly into the Hugging Face repository.
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
*© 2024 Aptech Limited — For Educational Use*
