# Session 05: Policy Optimization and Actor-Critic Methods

Welcome to **Part 4** of the Applied Machine Learning Using Python curriculum! 

In Session 3, we explored **Value-Based** Reinforcement Learning (Q-Learning, SARSA), where the agent learns the *value* of being in a state and taking an action. In this session, we transition to modern Deep Reinforcement Learning by exploring **Policy-Based** and **Actor-Critic** methods. Instead of calculating intermediate values to guess the best action, we will teach neural networks to directly output the best action to take!

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. List and explain different policy optimization techniques.
2. Understand the Policy Gradient Theorem and implement the REINFORCE algorithm.
3. Describe Actor-Critic architectures (and why combining Value and Policy methods is powerful).
4. Implement continuous and discrete control agents using TensorFlow/Keras and Gymnasium.
5. Understand modern advancements like A2C, TRPO, and PPO.

---

## 📖 Part 1: Why Policy Optimization?

### Value-Based vs. Policy-Based RL

| Feature | Value-Based (e.g., Q-Learning) | Policy-Based (e.g., REINFORCE) |
|---------|--------------------------------|--------------------------------|
| **What is learned?** | $Q(s, a)$ values | The policy $\pi_\theta(a|s)$ directly |
| **Action Space** | Must be small and discrete | Can be continuous (e.g., steering a car) |
| **Policy Type** | Generally Deterministic (greedy) | Can be Stochastic (probabilities) |
| **Convergence** | Not guaranteed to converge | Better theoretical convergence guarantees |

### The Limitations of Q-Learning
While Q-Learning is excellent for grid worlds and basic games, it fails in the real world:
1. **Continuous Action Spaces**: You can't compute `max_a Q(s,a)` if the actions are continuous (e.g., applying 12.345% pressure to a robot's brake).
2. **Stochastic Policies**: In games like Rock-Paper-Scissors, the optimal policy is to be perfectly random (stochastic). Q-Learning struggles to learn true randomness.

To solve these, we directly optimize the policy using **Neural Networks**.

---

## 📖 Part 2: The Policy Gradient Theorem and REINFORCE

### The Goal
Our goal is to find policy parameters $\theta$ (the weights of our neural network) that maximize the **expected return** $J(\theta)$ (total rewards accumulated over time).

### The Policy Gradient Theorem
We use gradient ascent to update the weights:
$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

The math simplifies beautifully into a highly intuitive concept:
> **"If an action resulted in a high reward, increase its probability. If it resulted in a low reward, decrease its probability."**

### The REINFORCE Algorithm (Monte Carlo Policy Gradient)
REINFORCE is the most fundamental policy gradient algorithm. 
1. **Play an episode:** Generate a trajectory $(s_0, a_0, r_1, s_1, ...)$ using current policy $\pi_\theta$.
2. **Calculate Returns:** For every step $t$, calculate the discounted cumulative return $G_t$.
3. **Update Network:** For every step, compute the gradient of the log-probability of the action taken, multiplied by the return $G_t$.

**The Problem with REINFORCE:**
Because it relies on full episodes (Monte Carlo), the returns $G_t$ vary wildly from episode to episode. This causes **high variance**, making training incredibly slow and unstable.

---

## 📖 Part 3: Actor-Critic Architectures

How do we fix the high variance of REINFORCE? We introduce a **Baseline** to determine if an action was *better than average*.

Enter **Actor-Critic**. It combines the best of both Policy-Based and Value-Based RL!

### The Two Neural Networks:
1. 🎭 **The Actor (Policy Network)**: Decides *what to do*. It observes the state and outputs action probabilities $\pi(a|s)$. 
2. 👨‍⚖️ **The Critic (Value Network)**: Evaluates *how good the action was*. It observes the state and outputs the expected value $V(s)$.

### How it Works (The Advantage Function)
Instead of updating the Actor based on raw rewards $G_t$, we update it based on the **Advantage** $A(s, a)$.

> $A(s, a) =$ Reward Received $-$ Expected Value (The Critic's guess)

- If $A > 0$: The action was better than the Critic expected! (Increase probability)
- If $A < 0$: The action was worse than the Critic expected! (Decrease probability)

### A2C (Advantage Actor-Critic)
A2C is the standard implementation. The Critic uses Temporal Difference (TD) errors to calculate the Advantage instantly at every step, meaning we don't have to wait until the end of the episode to learn (unlike REINFORCE).

---

## 📖 Part 4: Modern Policy Optimization (Advanced)

While A2C is excellent, it is still highly sensitive to hyperparameter tuning and learning rates. If the learning rate is too high, the policy network takes an awkwardly large step and "forgets" how to play entirely ("falling off a cliff").

### TRPO (Trust Region Policy Optimization)
Introduced in 2015, TRPO solves this by mathematically restricting the policy update size. It uses a "Trust Region" to ensure the new policy $\pi_{new}$ doesn't stray too far from the old policy $\pi_{old}$. However, TRPO's math is highly complex and computationally expensive.

### PPO (Proximal Policy Optimization)
Released by OpenAI in 2017, PPO is arguably the **most popular RL algorithm in the world today** (used to train ChatGPT via RLHF!). 
- It achieves the stable updates of TRPO but uses much simpler math.
- It uses a "Clipped Objective Function" that literally *clips* (limits) the update if the ratio between the new policy and old policy exceeds a certain threshold (usually 0.2).
- **Why is PPO great?** It is simple to implement, sample efficient, and extremely robust to hyperparameter choices.

---

## 🚀 Hands-On: Session Code Files

In the `code/` directory, you will find modern Deep RL implementations using `TensorFlow/Keras` and `Gymnasium`:

1. **`01_reinforce_cartpole.py`**  
   Implements pure Policy Gradients. Watch a neural network learn to balance a pole on a moving cart by receiving a reward of +1 for every frame it doesn't fall over.
   
2. **`02_actor_critic_cartpole.py`**  
   Implements an Advantage Actor-Critic (A2C) model. This script directly compares how much faster the agent learns when the Critic provides a baseline for the Actor.

---

## 🛠️ Environment Prerequisites

Because we are working with neural networks and physics engines, ensure you have activated your virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Ensure `tensorflow` and `gymnasium` are installed via your `requirements.txt`.

---
  *© 2024 Aptech Limited — For Educational Use*
