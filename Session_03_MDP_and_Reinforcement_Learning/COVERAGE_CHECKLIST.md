# Session 03 — Topic Coverage Checklist

## TL4 Topics (Part 1 — MDP & Reinforcement Learning Fundamentals — 2 hours)

| # | Required Topic | Covered In | Status |
|---|----------------|------------|--------|
| 1 | Define Reinforcement Learning and its key components | README Part 1 (agent, env, reward, policy), Notebook §1-2 | ✅ |
| 2 | Explain Markov Decision Processes (MDPs) | README Part 2 (tuple definition, Markov property), Notebook §3 | ✅ |
| 3 | Describe the Bellman Equation and value functions | README Part 2 (V(s) and Q(s,a) equations), Notebook §3 | ✅ |
| 4 | Implement Q-Learning algorithm | README Part 3, `code/01_q_learning_gridworld.py`, Notebook §4 | ✅ |
| 5 | Explain ε-greedy exploration strategy | README Part 3 (epsilon-greedy), `code/03_multi_armed_bandit.py` | ✅ |
| 6 | Distinguish RL from supervised/unsupervised learning | README Part 1 (comparison table), Notebook §1 | ✅ |

## TL5 Topics (Part 2 — Advanced RL Methods — 2 hours)

| # | Required Topic | Covered In | Status |
|---|----------------|------------|--------|
| 1 | Implement SARSA (on-policy learning) | README Part 4, `code/02_sarsa_comparison.py`, Notebook §5 | ✅ |
| 2 | Compare model-free RL methods (Q-Learning vs SARSA) | README Part 4 (table), `code/02_sarsa_comparison.py` (Cliff Walking) | ✅ |
| 3 | Explain Policy Gradient methods | README Part 5 (REINFORCE), Notebook §6 | ✅ |
| 4 | Describe Multi-Armed Bandit problem | README Part 6, `code/03_multi_armed_bandit.py` | ✅ |
| 5 | Compare exploration strategies (ε-Greedy, UCB, Thompson) | README Part 6 (table), `code/03_multi_armed_bandit.py` (5 strategies) | ✅ |
| 6 | Identify real-world RL applications | README Part 7 (8 applications table), Notebook §7 | ✅ |

## Code Files

| File | Format | Content | Lines |
|------|--------|---------|-------|
| `code/01_q_learning_gridworld.py` | .py | Q-Learning on custom GridWorld, training + visualization | ~306 |
| `code/02_sarsa_comparison.py` | .py | SARSA vs Q-Learning vs Expected SARSA on Cliff Walking | ~370 |
| `code/03_multi_armed_bandit.py` | .py | 5 bandit strategies, ad selection application | ~380 |
| `notebooks/Session_03_RL_Fundamentals.ipynb` | .ipynb | Interactive: RL basics → MDP → Q-Learning → SARSA → Bandits | ~400 cells |

## Exercises

| Level | Count | Topics |
|-------|-------|--------|
| 🟢 Beginner | 3 | ε-greedy bandit, hand-trace Q-Learning, GridWorld modification |
| 🟡 Intermediate | 2 | Q-Learning vs SARSA comparison, hyperparameter sensitivity |
| 🔴 Advanced | 2 | UCB exploration implementation, Policy Gradient on custom env |

## Portfolio Component
- **RL Agent Visualization Dashboard** — Interactive Gradio/Streamlit app with algorithm comparison, adjustable hyperparameters, and deployment guide for Hugging Face Spaces
