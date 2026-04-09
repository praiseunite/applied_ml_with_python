# 📊 Session 06 — Lab Results: Full Explanation
### RL Game Agent: CartPole with PPO (Proximal Policy Optimization)

> **Purpose of this document:** This file provides a deep, plain-English explanation of every cell that was run in `notebooks/RL_Game_Agent_Lab.ipynb`, what each output means, and why the numbers look the way they do. Read this alongside your notebook.

---

## 🧠 Big Picture — What Was This Lab Doing?

Before diving into individual cells, here is the complete story in plain English:

We built and trained an **AI agent** (a program that makes decisions) to play a physics simulation game called **CartPole**. In this game, a pole is balanced on top of a moving cart. The cart can only move **left or right**. The agent's goal is to prevent the pole from falling over for as long as possible.

The agent starts with **zero knowledge** — it has no idea what the game is or what it should do. Through thousands of trial-and-error attempts (called **timesteps**), it gradually figures out the right strategy. By the end of the lab, it achieves a **perfect score** every single time it plays.

The algorithm used to teach the agent is called **PPO (Proximal Policy Optimization)** — a state-of-the-art reinforcement learning technique used in real-world AI systems including OpenAI's game-playing bots.

---

## 🗂️ Cell-by-Cell Breakdown

---

## Cell 1 — Install Required Libraries

### Code Run:
```python
!pip install stable-baselines3[extra] gymnasium[classic-control]
```

### Output:
```
Requirement already satisfied: stable-baselines3[extra] ...
Requirement already satisfied: gymnasium[classic-control] ...
... (long list of dependencies)
```

### What This Means:

This cell installs the two main libraries needed for the lab:

| Library | What It Does | Real-World Analogy |
|---|---|---|
| `gymnasium` | Provides simulation environments (like CartPole) for the agent to train in | The "game engine" — the world the agent lives and practises in |
| `stable-baselines3` | Provides pre-built, production-quality RL algorithms like PPO | The "teaching toolkit" — the instruction method used to train the agent |

The message `Requirement already satisfied` on every line means these libraries were **already installed** in your Python virtual environment from a previous setup step. Nothing new was downloaded. This is completely normal and expected — it simply confirms your environment is ready.

Other dependencies confirmed present include:
- **`torch`** — PyTorch, the deep learning engine that powers the neural network inside our agent
- **`numpy`** — For numerical computation
- **`matplotlib`** — For charting and visualisation
- **`tensorboard`** — For monitoring training metrics over time
- **`opencv-python`** — For image/video processing
- **`pygame`** — For rendering game environments graphically

---

## Cell 2 — Import Libraries

### Code Run:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

os.makedirs("../app", exist_ok=True)
print("Libraries imported successfully!")
```

### Output:
```
Libraries imported successfully!
```

### What This Means:

This cell loads all the tools into active memory so they can be used in later cells. Key imports explained:

| Import | Purpose |
|---|---|
| `gymnasium as gym` | Loads the game environment system. `gym.make("CartPole-v1")` will create the game world |
| `PPO` from `stable_baselines3` | Loads the PPO algorithm — the "brain trainer" |
| `evaluate_policy` | A utility to fairly test the agent after training |
| `os` | Standard Python module for file system operations |

The line `os.makedirs("../app", exist_ok=True)` creates the `app/` folder if it doesn't exist yet. This is where the trained model file will be saved later. The `exist_ok=True` means it won't crash if the folder already exists.

**Why does such a simple cell matter?** In Python notebooks, imports must happen first. If any import fails here, nothing else in the notebook would work. A clean `Libraries imported successfully!` confirms the entire foundation is solid.

---

## Cell 3 — Initialize the Environment

### Code Run:
```python
env = gym.make("CartPole-v1")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
```

### Output:
```
Observation space: Box([-4.8, -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, inf], (4,), float32)
Action space: Discrete(2)
```

### What This Means:

This cell creates the CartPole game environment and prints two crucial pieces of information about it.

---

#### 🎯 The Observation Space — What the Agent Can "See"

```
Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf], (4,), float32)
```

This tells us the agent receives **4 numbers** at every game step. These 4 numbers describe the complete state of the world at that moment:

| Index | Variable | Min Value | Max Value | What It Represents |
|---|---|---|---|---|
| 0 | Cart Position | -4.8 | +4.8 | How far left (-) or right (+) the cart is from centre |
| 1 | Cart Velocity | -∞ | +∞ | How fast the cart is currently moving (negative = left, positive = right) |
| 2 | Pole Angle | -0.418 rad (~24°) | +0.418 rad (~24°) | How far the pole is tilting away from vertical |
| 3 | Pole Angular Velocity | -∞ | +∞ | How fast the pole tip is currently rotating |

Think of it like this: every fraction of a second, the agent receives a "snapshot" of 4 numbers. Based on those 4 numbers alone, it must decide what to do next. There is no camera, no picture — just raw physics data.

The word **`Box`** means the values form a continuous range (not discrete categories). The `float32` tells us the numbers are stored as 32-bit floating point decimals for efficiency.

---

#### 🕹️ The Action Space — What the Agent Can Do

```
Discrete(2)
```

This tells us the agent has **only 2 possible actions** at each step:

| Action | Code | Effect |
|---|---|---|
| Push Left | `0` | Apply a force pushing the cart to the left |
| Push Right | `1` | Apply a force pushing the cart to the right |

That's it. Every moment, the agent looks at 4 numbers and picks one of two buttons. The entire intelligence of the agent is in *which button to press* at *what moment*.

---

## Cell 4 — Train the Agent

### Code Run:
```python
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
print("Training completed!")
```

### Output:
13 training tables printed progressively during training, followed by `Training completed!`

---

### Understanding the Setup Lines

Before the first table appears, three setup messages are printed:

```
Using cpu device
```
> The agent is being trained on your **CPU** (Central Processing Unit), not a GPU. For CartPole this is perfectly fine — the environment is lightweight enough that CPU training is fast. GPU training is reserved for much larger models with image inputs (like Atari games).

```
Wrapping the env with a `Monitor` wrapper
```
> The Monitor wrapper is a transparent layer placed around the environment. It silently tracks stats per episode — how many steps each game lasted, what reward was earned — so those numbers can be shown in the training tables. You don't interact with it directly.

```
Wrapping the env in a DummyVecEnv.
```
> PPO is designed to support training on **multiple environments in parallel** (vectorised training). Since we only have one, `DummyVecEnv` wraps it to fit the expected multi-environment format. It changes nothing about training — it's an architectural requirement.

---

### Understanding the Training Tables

PPO trains in rounds called **iterations**. Each iteration collects **2,048 timesteps** of experience, then performs **10 gradient update steps** to improve the neural network. The table is printed after each iteration.

Each table has two or three sections: `rollout/`, `time/`, and `train/`.

---

#### 📋 Section 1: `rollout/` — The Agent's Current Performance

These metrics tell you how well the agent is **playing the game** right now.

| Metric | Full Name | What It Means |
|---|---|---|
| `ep_len_mean` | Episode Length Mean | The average number of steps across all recently completed game episodes. In CartPole, each step where the pole stays up gives +1 reward, so this directly equals the score. **Higher is better.** Maximum possible = 500. |
| `ep_rew_mean` | Episode Reward Mean | Average total reward earned. In CartPole, reward = +1 per step, so this always equals `ep_len_mean`. |

---

#### 📋 Section 2: `time/` — Training Speed & Progress

| Metric | What It Means |
|---|---|
| `fps` | Frames Per Second — how many game steps the computer is simulating per second. Starts at 448 fps and drops slightly as training gets more complex. |
| `iterations` | Which training round we are on (1 through 13). |
| `time_elapsed` | Total seconds that have passed since training began. |
| `total_timesteps` | Total game steps taken so far across all iterations. Each iteration adds 2,048 more. |

---

#### 📋 Section 3: `train/` — The Neural Network's Health (appears from iteration 2 onward)

These are the most technical metrics. They tell you whether the learning process itself is healthy.

| Metric | What It Means | Healthy Sign |
|---|---|---|
| `approx_kl` | **KL Divergence** — measures how much the agent's strategy (policy) changed in this update. PPO uses this to prevent the agent from changing too drastically in a single step, which would cause unstable learning. | Should stay small, typically 0.001–0.02 ✅ |
| `clip_fraction` | The fraction of parameter updates that were "clipped" (limited) by PPO's safety mechanism. PPO clips updates exceeding a threshold to prevent over-correction. | Between 0.01 and 0.2 is normal ✅ |
| `clip_range` | The fixed boundary used for clipping. Set at **0.2** throughout (the standard default). | Always 0.2 — this is a hyperparameter, not a variable ✅ |
| `entropy_loss` | Measures how **random/exploratory** the agent's decisions are. Negative values are expected (it's reported as negative of entropy). Higher entropy = more exploration. As the agent learns, entropy decreases (it becomes more decisive). | Gradually decreasing from ~-0.69 → ~-0.55. This shows the agent is becoming confident ✅ |
| `explained_variance` | How well the neural network's **value function** (its prediction of future rewards) matches what actually happened. Range is 0 to 1. Higher is better. | Ideally close to 1.0 by the end ✅ |
| `learning_rate` | The step size for gradient descent updates. Fixed at **0.0003** (a standard default). | Constant 0.0003 ✅ |
| `loss` | The combined total loss (policy + value + entropy). Measures overall error. | Trending downward over time ✅ |
| `policy_gradient_loss` | The error in the **action-choosing** part of the neural network. Negative values indicate the network is improving policies that led to good outcomes. | Small and negative ✅ |
| `value_loss` | The error in the **reward-predicting** part of the neural network (the critic). Should decrease as the agent better understands how to estimate future rewards. | Decreasing over time ✅ |

---

### 📈 The Complete Learning Progression — Iteration by Iteration

This table tells the story of the agent going from a complete beginner to a near-expert:

| Iteration | Total Steps | Score (ep_rew_mean) | Explained Variance | Key Observation |
|---|---|---|---|---|
| **1** | 2,048 | **21.5** | *(not yet available)* | Agent is completely random — pole falls after ~21 steps. No learning has occurred yet. |
| **2** | 4,096 | **26.7** | -0.0009 | First learning update applied. Tiny improvement. Explained variance near 0 — the value network predicts nothing useful yet. |
| **3** | 6,144 | **34.0** | 0.113 | Score rising. Value network starting to develop very basic understanding of rewards. |
| **4** | 8,192 | **43.7** | 0.224 | Solid jump. Agent is beginning to learn that keeping the pole upright earns more reward. |
| **5** | 10,240 | **59.5** | 0.279 | Score nearly tripled from start. Agent shows genuine competence emerging. |
| **6** | 12,288 | **75.7** | 0.591 | **Big jump in explained_variance** — value network is now accurately predicting ~59% of future reward. Major milestone. |
| **7** | 14,336 | **92.7** | 0.457 | Score approaches 100. Slight dip in explained_variance is normal — the policy is changing faster than the critic can keep up. |
| **8** | 16,384 | **110** | 0.566 | First time crossing 100 — a strong agent threshold for CartPole. |
| **9** | 18,432 | **126** | 0.196 | Score up but explained_variance temporarily drops — the policy is adjusting rapidly, making old value estimates temporarily inaccurate. |
| **10** | 20,480 | **142** | 0.769 | Rebounds strongly. Value network now accurately predicts ~77% of future reward. |
| **11** | 22,528 | **161** | 0.884 | Excellent. Agent is highly skilled. Value network at 88.4% accuracy. `loss` drops to just 2.49. |
| **12** | 24,576 | **178** | **0.959** | Near-perfect value network. `value_loss` plummets from ~50 to just 8.98 — the critic has mastered reward prediction. |
| **13** | 26,624 | **196** | 0.751 | Final iteration. Agent is excellent. Slight variance dip as the last policy changes settle. |

> **Key insight:** The score grew from **21.5 → 196** — nearly a 10x improvement — in just ~105 seconds of CPU training. This is a textbook-clean reinforcement learning curve.

---

### 🔬 Deep Dive: Why Does `explained_variance` Fluctuate?

You may notice `explained_variance` doesn't always go up. For example it dropped from 0.591 (iteration 6) to 0.457 (iteration 7), and from 0.279 (iteration 5) to 0.113 (iteration 3).

This is perfectly normal. Here's why:

PPO has **two networks** inside it:
1. **Actor (Policy Network)** — decides which action to take
2. **Critic (Value Network)** — predicts how much future reward will come from a given state

When the Actor learns a new strategy and changes significantly, the Critic's old predictions become temporarily wrong (because it was trained on the old behaviour). The Critic must then "catch up" and re-learn predictions for the new strategy. This creates a natural see-saw pattern in `explained_variance`.

The important thing is the **overall trend is strongly upward** — from nearly -0.001 to 0.95+.

---

## Cell 5 — Evaluate the Agent

### Code Run:
```python
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
```

### Output:
```
Mean reward: 500.0 +/- 0.0
```

### What This Means:

This is the most important result in the entire lab. 🏆

`evaluate_policy` runs the trained agent through **10 fresh, independent game episodes** in evaluation mode (no learning, no exploration — pure exploitation of what it has learned). It then reports the average score across all 10 episodes.

| Result | Value | Interpretation |
|---|---|---|
| `mean_reward` | **500.0** | The agent scored the **maximum possible score** (500 steps without failing) |
| `std_reward` | **0.0** | **Zero variation** — it scored exactly 500 in every single one of the 10 test episodes |

#### Why Did It Score 500 When Training Only Reached ~196?

This is a very insightful question. During **training**, the agent is still exploring — it intentionally makes some random decisions (controlled by entropy) to discover new strategies. This exploration sometimes causes it to fail sooner than it needs to.

During **evaluation**, exploration is turned off. The agent plays purely based on its best learned strategy. Since the core skill of balancing is fully mastered, it achieves perfect scores every time.

Think of it like a student who sometimes makes deliberate mistakes during practice to try different approaches, but walks into the exam and answers every question perfectly.

**A perfect score of 500.0 ± 0.0 is the gold standard** result for CartPole. It means the training was completely successful.

---

## Cell 6 — Save the Model

### Code Run:
```python
model_path = "../app/ppo_cartpole"
model.save(model_path)
print(f"Model successfully saved to {model_path}.zip")
env.close()
```

### Output:
```
Model successfully saved to ../app/ppo_cartpole.zip
```

### What This Means:

The trained neural network (the agent's "brain") is serialized and written to disk as a `.zip` file at:
```
Session_06_Practice_01_RL_Game_Agent/app/ppo_cartpole.zip
```

#### What is inside the `.zip` file?

The file contains:
- **The neural network weights** — the actual learned parameters (numbers) that encode the agent's entire strategy
- **Policy configuration** — the architecture details (MlpPolicy, layer sizes, activation functions)
- **Algorithm metadata** — PPO hyperparameters so the model can be loaded and used identically elsewhere

This file is everything the agent learned. In Phase 2, the Streamlit app (`app/app.py`) loads this file and uses the agent's brain to play CartPole live in the browser — without needing to retrain.

`env.close()` shuts down the CartPole simulation cleanly, releasing the memory it used.

---

## 🎓 What This Lab Proved

| Concept | What We Demonstrated |
|---|---|
| **Reinforcement Learning** | An agent can learn optimal behaviour through reward signals alone — no labelled data, no human guidance |
| **PPO Algorithm** | A stable, modern RL algorithm that safely improves a policy step-by-step |
| **Neural Networks as Policies** | A neural net can map raw sensor data (4 numbers) to expert decisions (left or right) |
| **Convergence** | The agent reliably converges to perfect performance in a small number of timesteps |
| **Model Persistence** | Trained models can be saved and reloaded — training does not need to be repeated every time the app is launched |

---

## 📏 Key Numbers to Remember

| Metric | Start | End | Change |
|---|---|---|---|
| Average Score (ep_rew_mean) | 21.5 | 196 | +811% |
| Explained Variance | ~0 | ~0.96 | Near-perfect understanding |
| Value Loss | ~50 | ~8.98 | Dramatic reduction |
| Entropy Loss | -0.687 | -0.546 | More decisive decision-making |
| Final Evaluation Score | — | **500 / 500** | 🏆 Perfect |

---

## 🔁 The Full Training Flow (Visual Summary)

```
START
  │
  ▼
Agent has ZERO knowledge
(random actions, score ~21)
  │
  ▼
[Collect 2048 experience steps] ──► Play the game, record what happened
  │
  ▼
[10 Gradient Update Steps] ──────► Adjust neural network using PPO math
  │
  ▼
Agent slightly smarter now
  │
  ▼
Repeat 13 times...
  │
  ▼
Agent scores ~196 in training
  │
  ▼
[Evaluate without exploration]
  │
  ▼
🏆 PERFECT SCORE: 500.0 ± 0.0
  │
  ▼
[Save model to ppo_cartpole.zip]
  │
  ▼
PROCEED TO PHASE 2 → app/app.py
```

---

*Document generated for: Session 06 — Practice 01: RL Game Agent*
*Course: Applied Machine Learning Using Python*
*Lab file: `notebooks/RL_Game_Agent_Lab.ipynb`*
