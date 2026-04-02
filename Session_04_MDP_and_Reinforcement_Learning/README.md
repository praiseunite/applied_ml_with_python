# Session 04: Markov Decision Process & Reinforcement Learning

<p align="center">
  <img src="https://img.shields.io/badge/Duration-4%20Hours-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TL4%20+%20TL5-Sessions%204--5-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Level-All%20Levels-green?style=for-the-badge" />
</p>

> **Covers**: TL4 (Session 4) and TL5 (Session 5) — 4 hours total
>
> **Book Reference**: Applied Machine Learning Using Python — Session 3

---

## 🎯 Learning Objectives

| # | Objective | Level |
|---|-----------|-------|
| 1 | Define Reinforcement Learning and its key components | 🟢 Beginner |
| 2 | Explain Markov Decision Processes (MDPs) | 🟢 Beginner |
| 3 | Implement Q-Learning and SARSA algorithms | 🟡 Intermediate |
| 4 | Compare model-free and model-based RL methods | 🟡 Intermediate |
| 5 | Implement Policy Gradient and Actor-Critic methods | 🔴 Advanced |
| 6 | Build an RL agent that learns to solve a real environment | 🔴 Advanced |

---

## 📋 Prerequisites

- Session 01 & 02 completed
- Understanding of probability and expected values
- Basic Python programming and NumPy

---

## Table of Contents

- [Part 1: What is Reinforcement Learning?](#part-1-what-is-reinforcement-learning)
- [Part 2: Markov Decision Processes](#part-2-markov-decision-processes-mdp)
- [Part 3: Q-Learning (Value-Based RL)](#part-3-q-learning-value-based-rl)
- [Part 4: SARSA (On-Policy Learning)](#part-4-sarsa-on-policy-learning)
- [Part 5: Policy Gradient Methods](#part-5-policy-gradient-methods)
- [Part 6: Multi-Armed Bandit Problem](#part-6-multi-armed-bandit-problem)
- [Part 7: Real-World RL Applications](#part-7-real-world-rl-applications)
- [Hands-On Lab](#-hands-on-lab)
- [Portfolio Task](#-portfolio-task)

---

## Part 1: What is Reinforcement Learning?

### 🟢 Beginner Level

**Reinforcement Learning (RL)** is a type of ML where an **agent** learns to make decisions by interacting with an **environment**. The agent receives **rewards** or **penalties** based on its actions and learns to maximize total reward over time.

#### The Key Difference from Other ML Types

| Type | Data | Learning Signal |
|------|------|----------------|
| **Supervised Learning** | Labeled data (X → Y) | Correct answers provided |
| **Unsupervised Learning** | Unlabeled data (X only) | No feedback, find structure |
| **Reinforcement Learning** | Sequential interactions | Delayed reward signals |

#### Real-World Analogy

Think of training a dog:
- 🐕 **Agent**: The dog
- 🌍 **Environment**: Your house
- 🎯 **Actions**: Sit, stay, fetch, bark
- 🍖 **Reward**: Treats for good behavior (+1)
- 🚫 **Penalty**: Scolding for bad behavior (-1)
- 📋 **Policy**: The dog's learned behavior (what to do in each situation)

The dog doesn't know the rules initially — it **explores**, tries actions, and learns from consequences.

#### RL Framework

```
            ┌──────────────────┐
            │   ENVIRONMENT    │
            │                  │
    Action  │   State: s_t     │  Reward: r_t
   ────────►│   Reward: r_t    │◄────────
            │   Next State:    │
            │     s_{t+1}      │
            └────────┬─────────┘
                     │
                     │ Observation
                     ▼
            ┌──────────────────┐
            │      AGENT       │
            │                  │
            │   Policy: π(s)   │
            │   "What action   │
            │    to take in    │
            │    state s"      │
            └──────────────────┘
```

```python
"""
RL Framework — The Core Loop
Every RL algorithm follows this pattern.
"""
def rl_loop(agent, environment, n_episodes=100):
    """The fundamental RL training loop."""
    for episode in range(n_episodes):
        state = environment.reset()        # Start fresh
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)         # Use policy
            next_state, reward, done = environment.step(action)  # Execute
            agent.learn(state, action, reward, next_state)       # Update
            state = next_state
            total_reward += reward

        print(f"Episode {episode}: Reward = {total_reward}")
```

#### Key Components

| Component | Symbol | Description | Example (Chess) |
|-----------|--------|-------------|-----------------|
| **State** | s | Current situation | Board position |
| **Action** | a | What the agent does | Move a piece |
| **Reward** | r | Immediate feedback | +1 win, -1 lose, 0 otherwise |
| **Policy** | π(s) | Strategy: state → action | "In this position, move queen" |
| **Value Function** | V(s) | Expected future reward from state s | "This board position is favorable" |
| **Q-Function** | Q(s,a) | Expected reward for action a in state s | "Moving knight here yields +0.7" |
| **Discount Factor** | γ | How much to value future vs immediate rewards | γ=0.99 (patient), γ=0.1 (greedy) |

---

## Part 2: Markov Decision Processes (MDP)

### 🟢 Beginner Level

An **MDP** is the mathematical framework that RL is built on. It formalizes the problem the agent is trying to solve.

#### MDP Definition

An MDP is defined by a tuple **(S, A, P, R, γ)** where:

| Symbol | Name | Meaning |
|--------|------|---------|
| **S** | State space | All possible situations |
| **A** | Action space | All possible actions |
| **P(s'|s,a)** | Transition probability | Probability of reaching state s' from state s after action a |
| **R(s,a)** | Reward function | Immediate reward for taking action a in state s |
| **γ** | Discount factor | How much to value future rewards (0 ≤ γ ≤ 1) |

#### The Markov Property

> **"The future depends only on the present, not the past."**
>
> P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, s_1, a_1, ..., s_t, a_t)

This means: **the current state contains all information needed** to make the best decision. You don't need to remember history.

#### Simple Example: GridWorld

```
┌─────┬─────┬─────┬─────┐
│  S  │     │     │ +1  │  S = Start
│     │     │     │GOAL │  X = Wall
├─────┼─────┼─────┼─────┤  +1 = Goal (reward)
│     │  X  │     │ -1  │  -1 = Trap (penalty)
│     │WALL │     │TRAP │
├─────┼─────┼─────┼─────┤
│     │     │     │     │
│     │     │     │     │
└─────┴─────┴─────┴─────┘

States: Each cell (12 total, minus wall = 11)
Actions: {UP, DOWN, LEFT, RIGHT}
Rewards: +1 at goal, -1 at trap, -0.04 per step
```

### 🟡 Intermediate Level

#### Bellman Equation

The **Bellman Equation** is the fundamental equation of RL. It expresses the value of a state as the immediate reward plus the discounted value of the next state:

**State Value Function:**
```
V(s) = max_a [ R(s,a) + γ · Σ P(s'|s,a) · V(s') ]
```

**Action Value Function (Q-function):**
```
Q(s,a) = R(s,a) + γ · Σ P(s'|s,a) · max_a' Q(s', a')
```

> **In plain English**: "The value of being in a state = the best reward I can get now + how good the next state will be."

---

## Part 3: Q-Learning (Value-Based RL)

### 🟢 Beginner Level

**Q-Learning** is the most popular RL algorithm. It learns the **Q-table** — a lookup table that tells the agent what action to take in each state.

#### Q-Learning Update Rule

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') – Q(s, a)]
                         └────────── TD target ──────────┘
                    └──────────────── TD error ──────────────┘
```

Where:
- **α** = learning rate (how fast to learn)
- **γ** = discount factor (how much to value future)
- **r** = immediate reward
- **max Q(s', a')** = best possible value from next state

### 🟡 Intermediate Level — Implementation

```python
"""
Q-Learning: Frozen Lake Environment
The agent must cross a frozen lake (4x4 grid) without falling
into holes, reaching the goal.

States: 16 (4x4 grid)
Actions: 4 (LEFT, DOWN, RIGHT, UP)
Reward: +1 for reaching goal, 0 otherwise
"""
import numpy as np

class FrozenLake:
    """Simple 4x4 FrozenLake environment."""

    def __init__(self):
        self.grid_size = 4
        self.n_states = 16
        self.n_actions = 4  # LEFT, DOWN, RIGHT, UP
        self.holes = {5, 7, 11, 12}  # Hole positions
        self.goal = 15
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.grid_size)

        # Move
        if action == 0: col = max(0, col - 1)        # LEFT
        elif action == 1: row = min(3, row + 1)       # DOWN
        elif action == 2: col = min(3, col + 1)       # RIGHT
        elif action == 3: row = max(0, row - 1)       # UP

        self.state = row * self.grid_size + col

        # Check outcome
        if self.state in self.holes:
            return self.state, -1.0, True   # Fell in hole
        elif self.state == self.goal:
            return self.state, 1.0, True    # Reached goal!
        else:
            return self.state, -0.01, False # Small step penalty

def q_learning(env, n_episodes=5000, alpha=0.1, gamma=0.99, epsilon=1.0,
               epsilon_decay=0.9995, epsilon_min=0.01):
    """Train a Q-Learning agent."""
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)  # Explore
            else:
                action = np.argmax(Q[state])               # Exploit

            next_state, reward, done = env.step(action)

            # Q-Learning update (off-policy: uses max)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, rewards_per_episode

# Train the agent
env = FrozenLake()
Q_table, rewards = q_learning(env)

# Results
print("Q-Learning Training Complete!")
print(f"Last 100 episodes avg reward: {np.mean(rewards[-100:]):.3f}")
print(f"Success rate (last 500): {sum(1 for r in rewards[-500:] if r > 0) / 500:.1%}")

# Show learned policy
actions = ['←', '↓', '→', '↑']
print("\nLearned Policy:")
for row in range(4):
    line = ""
    for col in range(4):
        s = row * 4 + col
        if s in {5, 7, 11, 12}:
            line += " ■ "
        elif s == 15:
            line += " ★ "
        else:
            line += f" {actions[np.argmax(Q_table[s])]} "
    print(line)
```

---

## Part 4: SARSA (On-Policy Learning)

### 🟡 Intermediate Level

**SARSA** (**S**tate-**A**ction-**R**eward-**S**tate-**A**ction) is similar to Q-Learning but uses the **actual next action** instead of the **best** next action.

#### Key Difference

| Algorithm | Update Uses | Behavior |
|-----------|------------|----------|
| **Q-Learning** (off-policy) | `max Q(s', a')` — best possible action | More optimistic, can learn from any policy |
| **SARSA** (on-policy) | `Q(s', a')` — actual next action taken | More conservative, learns the policy it follows |

```python
"""
SARSA Update Rule:
Q(s,a) ← Q(s,a) + α · [r + γ · Q(s', a') – Q(s,a)]
                                  ↑ actual next action (not max!)
"""
def sarsa(env, n_episodes=5000, alpha=0.1, gamma=0.99, epsilon=1.0,
          epsilon_decay=0.9995, epsilon_min=0.01):
    """Train a SARSA agent (on-policy)."""
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # Choose first action
        if np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state])

        while not done:
            next_state, reward, done = env.step(action)

            # Choose NEXT action (using current policy)
            if np.random.random() < epsilon:
                next_action = np.random.randint(env.n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # SARSA update: uses Q(s', a') not max Q(s', ·)
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, rewards_per_episode
```

---

## Part 5: Policy Gradient Methods

### 🔴 Advanced Level

Policy gradient methods directly learn the **policy** (π) instead of learning values and deriving a policy from them.

#### Why Policy Gradient?

| Feature | Value-Based (Q-Learning) | Policy Gradient |
|---------|-------------------------|-----------------|
| **Action space** | Works with discrete actions | Works with continuous actions too |
| **Policy type** | Deterministic (argmax Q) | Stochastic (probability distribution) |
| **Convergence** | Can oscillate | Smoother convergence |
| **Best for** | Small discrete action spaces | Large/continuous action spaces |

#### REINFORCE Algorithm (Monte Carlo Policy Gradient)

```python
"""
REINFORCE — Simplified Policy Gradient
Uses a simple neural-network-like approach to learn policy weights.
"""
import numpy as np

class PolicyGradientAgent:
    """Simple policy gradient with softmax policy."""

    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.theta = np.zeros((n_states, n_actions))  # Policy params

    def get_action_probs(self, state):
        """Softmax policy."""
        logits = self.theta[state]
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / exp_logits.sum()

    def choose_action(self, state):
        probs = self.get_action_probs(state)
        return np.random.choice(self.n_actions, p=probs)

    def update(self, trajectory):
        """Update policy using REINFORCE (full episode)."""
        states, actions, rewards = zip(*trajectory)

        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalize returns (baseline)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy parameters
        for t, (s, a, G_t) in enumerate(zip(states, actions, returns)):
            probs = self.get_action_probs(s)
            # Gradient: increase prob of actions that led to high returns
            for a_i in range(self.n_actions):
                if a_i == a:
                    self.theta[s, a_i] += self.lr * G_t * (1 - probs[a_i])
                else:
                    self.theta[s, a_i] -= self.lr * G_t * probs[a_i]
```

---

## Part 6: Multi-Armed Bandit Problem

### 🟢 Beginner Level

The **Multi-Armed Bandit** is the simplest RL problem — choosing between slot machines to maximize total reward.

#### Exploration vs Exploitation Dilemma

> **Explore**: Try new actions to discover potentially better rewards
> **Exploit**: Use the best known action to maximize reward
> The challenge: **how to balance both?**

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **ε-Greedy** | Random action with probability ε, best action otherwise | Simple, works well |
| **UCB (Upper Confidence Bound)** | Choose action with highest `Q(a) + c·√(ln(t)/N(a))` | When you want principled exploration |
| **Thompson Sampling** | Sample from posterior distribution of each arm | Bayesian approach, often best |

---

## Part 7: Real-World RL Applications

| Application | Company | RL Method | Impact |
|-------------|---------|-----------|--------|
| **Game Playing** | DeepMind (AlphaGo) | Monte Carlo Tree Search + RL | Beat world champion at Go |
| **Robotics** | Boston Dynamics | Policy Gradient | Robot locomotion and control |
| **Recommendations** | Netflix, YouTube | Contextual Bandits | Personalized content ranking |
| **Trading** | Hedge Funds | Deep Q-Networks | Automated trading strategies |
| **Resource Management** | Google DeepMind | DQN | 40% reduction in data center cooling energy |
| **Autonomous Driving** | Waymo, Tesla | Actor-Critic | Lane changing, parking decisions |
| **Healthcare** | IBM | Bandits | Dynamic treatment regimens |
| **NLP** | OpenAI (ChatGPT) | RLHF (RL from Human Feedback) | Aligns LLMs with human preferences |

---

## 💻 Hands-On Lab

### Lab 1: Q-Learning on FrozenLake (45 min)
```bash
cd Session_03_MDP_and_Reinforcement_Learning/code
python 01_q_learning_gridworld.py
```

### Lab 2: SARSA vs Q-Learning Comparison (30 min)
```bash
python 02_sarsa_comparison.py
```

### Lab 3: Multi-Armed Bandit Strategies (30 min)
```bash
python 03_multi_armed_bandit.py
```

---

## ✍️ Exercises

See [exercises/exercises.md](exercises/exercises.md)

| Level | Exercise | Topic |
|-------|----------|-------|
| 🟢 | 3.1 | Implement ε-greedy bandit |
| 🟢 | 3.2 | Trace Q-Learning by hand |
| 🟡 | 3.3 | Q-Learning vs SARSA comparison |
| 🟡 | 3.4 | Tune hyperparameters (α, γ, ε) |
| 🔴 | 3.5 | Implement UCB exploration |
| 🔴 | 3.6 | Policy Gradient on custom environment |

---

## 📊 Portfolio Task

Build an **RL Agent Visualization Dashboard** — See `portfolio/portfolio_component.md`

---

## 📚 Further Reading

- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd Ed.) — **The RL Bible** ([free online](http://incompleteideas.net/book/the-book-2nd.html))
- Silver, D. (2015). UCL RL Lecture Series — [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- OpenAI Spinning Up — [spinningup.openai.com](https://spinningup.openai.com/)

---

## [⬅️ Session 02](../Session_02_Advanced_Clustering/) | [🏠 Home](../README.md) | [Practice Session 1 ➡️](../Practice_Session_01/)
