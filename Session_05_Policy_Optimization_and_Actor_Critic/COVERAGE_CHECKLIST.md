# Session 04 — Topic Coverage Checklist

## TL5 Extended Topics (Policy Optimization and Actor-Critic — 4 hours)

| # | Required Topic | Covered In | Status |
|---|----------------|------------|--------|
| 1 | List different policy optimization techniques | README Part 4 (REINFORCE, TRPO, PPO), Notebook §1 | ✅ |
| 2 | Explain the objective of Policy Gradient methods | README Part 2 (Policy Gradient Theorem), Notebook §2 | ✅ |
| 3 | Evaluate the limitations of Value-Based vs Policy-Based RL | README Part 1 (Continuous vs Discrete), Notebook §1 | ✅ |
| 4 | Explain the High Variance problem in REINFORCE | README Part 2, Notebook §4, Ex 4.3 (Math proof) | ✅ |
| 5 | Describe Actor Critic methods | README Part 3, Notebook §5, `02_actor_critic_cartpole.py` | ✅ |
| 6 | Differentiate between the Actor and the Critic roles | README Part 3 (Actor = Policy, Critic = Value Baseline) | ✅ |
| 7 | Implement a neural network policy generator (TensorFlow/Keras) | `code/01_reinforce_cartpole.py`, Ex 4.2 | ✅ |
| 8 | Formulate policy gradient clipping and safety measures | README Part 4 (PPO/TRPO limit discussions), Ex 4.6 | ✅ |

## Code Files

| File | Format | Content | Lines |
|------|--------|---------|-------|
| `code/01_reinforce_cartpole.py` | .py | Pure Monte Carlo Policy Gradient on CartPole using Keras | ~140 |
| `code/02_actor_critic_cartpole.py` | .py | Advantage Actor-Critic (A2C) on CartPole using Keras | ~160 |
| `notebooks/Session_04_Policy_Optimization.ipynb` | .ipynb | Interactive: Limitations → REINFORCE → Baselines → Actor-Critic | ~350 cells |

## Exercises

| Level | Count | Topics |
|-------|-------|--------|
| 🟢 Beginner | 2 | Calculating Returns manually, Building a Keras Policy Network |
| 🟡 Intermediate | 2 | Variance Reduction math proof, Categorical vs Custom PG Loss function |
| 🔴 Advanced | 2 | Continuous Action parameterization (Mu/Sigma), Gradient Clipping |

## Portfolio Component
- **Real-Time Actor-Critic Training Dashboard** — A live-streaming Gradio dashboard demonstrating MLOps monitoring of a training TensorFlow Deep RL agent, ready for Hugging Face Spaces deployment.
