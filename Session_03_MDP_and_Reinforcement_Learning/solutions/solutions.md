# Session 03 — Solutions

## 🟢 Solution 3.1: ε-Greedy Bandit

```python
"""
Solution 3.1: ε-Greedy Bandit
──────────────────────────────
A complete implementation of the ε-greedy multi-armed bandit strategy.
"""
import numpy as np
import matplotlib.pyplot as plt

class SimpleBandit:
    """A 5-armed bandit with known reward probabilities."""
    def __init__(self):
        self.true_probs = [0.1, 0.3, 0.7, 0.4, 0.2]
        self.n_arms = 5
        self.best_arm = 2

    def pull(self, arm):
        return float(np.random.random() < self.true_probs[arm])

bandit = SimpleBandit()
n_steps = 1000
epsilon = 0.1

Q = np.zeros(bandit.n_arms)
N = np.zeros(bandit.n_arms)
optimal_count = 0
reward_history = []
optimal_history = []

for t in range(n_steps):
    # ε-Greedy action selection
    if np.random.random() < epsilon:
        action = np.random.randint(bandit.n_arms)  # EXPLORE
    else:
        action = np.argmax(Q)                       # EXPLOIT

    reward = bandit.pull(action)

    # Incremental mean update
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]

    if action == bandit.best_arm:
        optimal_count += 1

    reward_history.append(reward)
    optimal_history.append(optimal_count / (t + 1))

# Results
print(f"Optimal arm selected: {optimal_count / n_steps:.1%}")
print(f"\nEstimated vs True values:")
for i in range(bandit.n_arms):
    print(f"  Arm {i}: Q̂ = {Q[i]:.3f}  (true = {bandit.true_probs[i]:.1f})"
          f"  pulled {int(N[i])} times")

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Bar chart: estimated vs true
x = np.arange(bandit.n_arms)
width = 0.35
ax1.bar(x - width/2, bandit.true_probs, width, label='True', color='#3498db')
ax1.bar(x + width/2, Q, width, label='Estimated', color='#e74c3c')
ax1.set_xlabel('Arm')
ax1.set_ylabel('Value')
ax1.set_title('True vs Estimated Values')
ax1.legend()
ax1.set_xticks(x)

# Optimal action rate over time
ax2.plot(optimal_history, linewidth=1, color='#2ecc71')
ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5,
            label='Expected (~90%)')
ax2.set_xlabel('Step')
ax2.set_ylabel('Optimal Action %')
ax2.set_title('Optimal Action Rate Over Time')
ax2.legend()

# Arm pull distribution
ax3.bar(x, N, color='#9b59b6')
ax3.set_xlabel('Arm')
ax3.set_ylabel('Times Pulled')
ax3.set_title('Arm Pull Distribution')
ax3.set_xticks(x)

plt.suptitle('ε-Greedy Bandit Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Expected Output**:
```
Optimal arm selected: ~88-93%

Estimated vs True values:
  Arm 0: Q̂ ≈ 0.100  (true = 0.1)  pulled ~20 times
  Arm 1: Q̂ ≈ 0.300  (true = 0.3)  pulled ~20 times
  Arm 2: Q̂ ≈ 0.700  (true = 0.7)  pulled ~900 times
  Arm 3: Q̂ ≈ 0.400  (true = 0.4)  pulled ~20 times
  Arm 4: Q̂ ≈ 0.200  (true = 0.2)  pulled ~20 times
```

**Key Learnings**:
- ε-greedy splits time: ~90% exploitation, ~10% exploration
- The exploratory 10% is split equally across ALL arms (including bad ones)
- Estimated Q-values converge to true values given enough samples
- The best arm gets the most pulls (exploitation) while others are pulled roughly equally (exploration)

---

## 🟢 Solution 3.2: Trace Q-Learning by Hand

### Part A: Q-Learning Updates

| Step | State | Action | Reward | Next State | Calculation | New Q(s,a) |
|------|-------|--------|--------|------------|-------------|------------|
| 1 | S0 | RIGHT | -1 | S1 | Q(S0,R) ← 0 + 0.5×[-1 + 0.9×max(0,0) - 0] = 0.5×(-1) | **-0.500** |
| 2 | S1 | RIGHT | +10 | S2 (done) | Q(S1,R) ← 0 + 0.5×[10 + 0.9×0 - 0] = 0.5×(10) | **5.000** |
| 3 | S0 | RIGHT | -1 | S1 | Q(S0,R) ← -0.5 + 0.5×[-1 + 0.9×max(0,5) - (-0.5)] = -0.5 + 0.5×(3.5+0.5) | **1.500** |
| 4 | S1 | LEFT | -1 | S0 | Q(S1,L) ← 0 + 0.5×[-1 + 0.9×max(1.5,0) - 0] = 0.5×(0.35) | **0.175** |
| 5 | S0 | RIGHT | -1 | S1 | Q(S0,R) ← 1.5 + 0.5×[-1 + 0.9×max(0.175,5) - 1.5] = 1.5 + 0.5×(2.0) | **2.500** |

### Part B: Final Q-Table

| State | LEFT | RIGHT |
|-------|------|-------|
| S0 | 0.000 | **2.500** |
| S1 | **0.175** | **5.000** |
| S2 | 0.000 | 0.000 |

### Part C: Implied Policy

The Q-table suggests: **S0 → RIGHT → S1 → RIGHT → S2**

This IS the optimal policy! But the Q-values haven't converged yet (with more episodes, Q(S0,RIGHT) would approach 8.0 = -1 + 0.9×(-1 + 0.9×10)). After only 5 steps, the agent has identified the correct path but hasn't learned the precise values yet.

### Part D: Verification Code

```python
import numpy as np

Q = np.zeros((3, 2))
alpha = 0.5
gamma = 0.9

# Step 1: S0, RIGHT(1), reward=-1, next=S1
Q[0, 1] = Q[0, 1] + alpha * (-1 + gamma * np.max(Q[1]) - Q[0, 1])
print(f"Step 1: Q(S0, RIGHT) = {Q[0, 1]:.4f}")  # -0.5000

# Step 2: S1, RIGHT(1), reward=+10, next=S2 (terminal)
Q[1, 1] = Q[1, 1] + alpha * (10 + gamma * 0 - Q[1, 1])  # Terminal: no future
print(f"Step 2: Q(S1, RIGHT) = {Q[1, 1]:.4f}")  # 5.0000

# Step 3: S0, RIGHT(1), reward=-1, next=S1
Q[0, 1] = Q[0, 1] + alpha * (-1 + gamma * np.max(Q[1]) - Q[0, 1])
print(f"Step 3: Q(S0, RIGHT) = {Q[0, 1]:.4f}")  # 1.5000

# Step 4: S1, LEFT(0), reward=-1, next=S0
Q[1, 0] = Q[1, 0] + alpha * (-1 + gamma * np.max(Q[0]) - Q[1, 0])
print(f"Step 4: Q(S1, LEFT)  = {Q[1, 0]:.4f}")  # 0.1750

# Step 5: S0, RIGHT(1), reward=-1, next=S1
Q[0, 1] = Q[0, 1] + alpha * (-1 + gamma * np.max(Q[1]) - Q[0, 1])
print(f"Step 5: Q(S0, RIGHT) = {Q[0, 1]:.4f}")  # 2.5000

print(f"\nFinal Q-table:\n{Q}")
```

**Key Learning**: By tracing updates manually, students see that Q-Learning:
1. Bootstraps — uses its own estimates to update
2. Propagates reward information backwards (goal reward spreads to earlier states)
3. Can learn the correct path even from small amounts of experience

---

## 🟡 Solution 3.4: Q-Learning vs SARSA on FrozenLake

```python
"""
Solution 3.4: Q-Learning vs SARSA Comparison
──────────────────────────────────────────────
Side-by-side comparison on the FrozenLake environment.
"""
import numpy as np
import matplotlib.pyplot as plt

class FrozenLake:
    def __init__(self):
        self.grid_size = 4
        self.n_states = 16
        self.n_actions = 4
        self.holes = {5, 7, 11, 12}
        self.goal = 15
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.grid_size)
        if action == 0: col = max(0, col - 1)
        elif action == 1: row = min(3, row + 1)
        elif action == 2: col = min(3, col + 1)
        elif action == 3: row = max(0, row - 1)
        self.state = row * self.grid_size + col

        if self.state in self.holes:
            return self.state, -1.0, True
        elif self.state == self.goal:
            return self.state, 1.0, True
        else:
            return self.state, -0.01, False


def q_learning(env, n_episodes=10000, alpha=0.1, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + gamma * best_next * (1 - done) - Q[state, action]
            )
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, rewards


def sarsa(env, n_episodes=10000, alpha=0.1, gamma=0.99,
          epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []

    for ep in range(n_episodes):
        state = env.reset()

        if np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state])

        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = env.step(action)

            if np.random.random() < epsilon:
                next_action = np.random.randint(env.n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # SARSA: uses actual next action
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] * (1 - done)
                - Q[state, action]
            )

            state = next_state
            action = next_action
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, rewards


# Train both
env_q = FrozenLake()
env_s = FrozenLake()

Q_q, rewards_q = q_learning(env_q)
Q_s, rewards_s = sarsa(env_s)


# Test both (greedy, no exploration)
def test_agent(env, Q, n_test=1000):
    successes, total_steps, total_reward = 0, 0, 0
    for _ in range(n_test):
        state = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        while not done and steps < 100:
            action = np.argmax(Q[state])
            state, reward, done = env.step(action)
            steps += 1
            ep_reward += reward
        if reward > 0:
            successes += 1
        total_steps += steps
        total_reward += ep_reward
    return successes / n_test, total_steps / n_test, total_reward / n_test


sr_q, steps_q, avg_r_q = test_agent(FrozenLake(), Q_q)
sr_s, steps_s, avg_r_s = test_agent(FrozenLake(), Q_s)

# Comparison Table
print(f"\n{'Metric':<35s} {'Q-Learning':>12s} {'SARSA':>12s}")
print("─" * 60)
print(f"{'Training success (last 1000):':<35s} "
      f"{sum(1 for r in rewards_q[-1000:] if r > 0)/1000:>11.1%} "
      f"{sum(1 for r in rewards_s[-1000:] if r > 0)/1000:>11.1%}")
print(f"{'Test success rate (1000 ep):':<35s} {sr_q:>11.1%} {sr_s:>11.1%}")
print(f"{'Avg steps to goal (test):':<35s} {steps_q:>12.1f} {steps_s:>12.1f}")
print(f"{'Avg reward (last 1000 train):':<35s} "
      f"{np.mean(rewards_q[-1000:]):>12.3f} "
      f"{np.mean(rewards_s[-1000:]):>12.3f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

window = 100
smooth_q = np.convolve(rewards_q, np.ones(window)/window, 'valid')
smooth_s = np.convolve(rewards_s, np.ones(window)/window, 'valid')

ax1.plot(smooth_q, label='Q-Learning', color='#e74c3c', linewidth=2)
ax1.plot(smooth_s, label='SARSA', color='#2ecc71', linewidth=2)
ax1.set_title('Learning Curves (Smoothed)', fontweight='bold')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show policies side by side
actions = ['←', '↓', '→', '↑']
for ax, Q, title in [(ax2, Q_q, 'Q-Learning'), (ax2, Q_s, 'SARSA')]:
    pass  # Policies visualized in console

# Print policies
for name, Q in [("Q-Learning", Q_q), ("SARSA", Q_s)]:
    print(f"\n{name} Policy:")
    for r in range(4):
        line = "  "
        for c in range(4):
            s = r * 4 + c
            if s in {5, 7, 11, 12}:
                line += " ■ "
            elif s == 15:
                line += " ★ "
            else:
                line += f" {actions[np.argmax(Q[s])]} "
        print(line)

ax2.bar(['Q-Learning', 'SARSA'], [sr_q * 100, sr_s * 100],
        color=['#e74c3c', '#2ecc71'])
ax2.set_ylabel('Success Rate (%)')
ax2.set_title('Test Success Rate', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Analysis** (sample answer):

> Q-Learning and SARSA learn similar policies on FrozenLake because the environment
> has deterministic transitions and the exploration noise is mainly in the ε-greedy
> action selection. Q-Learning typically converges slightly faster because it learns
> from the maximum future Q-value (optimistic), while SARSA learns from the actual
> next action (which includes exploratory moves). In environments with dangerous
> states near optimal paths (like Cliff Walking), SARSA would learn a safer strategy
> because it factors in the possibility of accidental exploration into its updates.
> On FrozenLake, both achieve similar success rates, but Q-Learning's off-policy
> nature makes it slightly more sample-efficient. SARSA's advantage emerges when
> training-time safety matters.

---

## 🟡 Solution 3.5: Hyperparameter Sensitivity (Partial)

```python
"""
Solution 3.5: Hyperparameter Sensitivity Analysis
───────────────────────────────────────────────────
Systematically explores α, γ, and ε-decay effects on Q-Learning.
"""
import numpy as np
import matplotlib.pyplot as plt

# Using FrozenLake from Solution 3.4 (omitted for brevity)
# ... (same class definition)

def run_experiment(alpha=0.1, gamma=0.99, epsilon_decay=0.9995,
                   n_episodes=5000, n_test=500):
    """Train Q-Learning and return success rate."""
    env = FrozenLake()
    Q = np.zeros((16, 4))
    epsilon = 1.0

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state])
            ns, r, done = env.step(action)
            Q[state, action] += alpha * (
                r + gamma * np.max(Q[ns]) * (1-done) - Q[state, action]
            )
            state = ns
        epsilon = max(0.01, epsilon * epsilon_decay)

    # Test
    successes = 0
    for _ in range(n_test):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = np.argmax(Q[state])
            state, r, done = env.step(action)
            steps += 1
        if r > 0:
            successes += 1
    return successes / n_test


# ─── Experiment 1: Learning Rate (α) ───
alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
alpha_results = []
for a in alphas:
    scores = [run_experiment(alpha=a) for _ in range(5)]
    alpha_results.append((np.mean(scores), np.std(scores)))
    print(f"α={a:.2f}: Success = {np.mean(scores):.1%} ± {np.std(scores):.1%}")

# ─── Experiment 2: Discount Factor (γ) ───
gammas = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
gamma_results = []
for g in gammas:
    scores = [run_experiment(gamma=g) for _ in range(5)]
    gamma_results.append((np.mean(scores), np.std(scores)))
    print(f"γ={g:.2f}: Success = {np.mean(scores):.1%} ± {np.std(scores):.1%}")

# ─── Experiment 3: Epsilon Decay ───
decays = [0.999, 0.9995, 0.9999, 1.0]
decay_results = []
for d in decays:
    scores = [run_experiment(epsilon_decay=d) for _ in range(5)]
    decay_results.append((np.mean(scores), np.std(scores)))
    print(f"ε_decay={d}: Success = {np.mean(scores):.1%} ± {np.std(scores):.1%}")

# ─── Visualization ───
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# α plot
means, stds = zip(*alpha_results)
ax1.errorbar(alphas, means, yerr=stds, fmt='o-', color='#3498db',
             linewidth=2, capsize=5, markersize=8)
ax1.set_xlabel('Learning Rate (α)')
ax1.set_ylabel('Success Rate')
ax1.set_title('Effect of Learning Rate', fontweight='bold')
ax1.grid(True, alpha=0.3)

# γ plot
means, stds = zip(*gamma_results)
ax2.errorbar(gammas, means, yerr=stds, fmt='s-', color='#2ecc71',
             linewidth=2, capsize=5, markersize=8)
ax2.set_xlabel('Discount Factor (γ)')
ax2.set_ylabel('Success Rate')
ax2.set_title('Effect of Discount Factor', fontweight='bold')
ax2.grid(True, alpha=0.3)

# ε-decay plot
means, stds = zip(*decay_results)
ax3.errorbar(range(len(decays)), means, yerr=stds, fmt='D-', color='#e74c3c',
             linewidth=2, capsize=5, markersize=8)
ax3.set_xticks(range(len(decays)))
ax3.set_xticklabels([str(d) for d in decays])
ax3.set_xlabel('Epsilon Decay')
ax3.set_ylabel('Success Rate')
ax3.set_title('Effect of Exploration Decay', fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Recommendations**:
- **α (learning rate)**: 0.1-0.3 works best. Too low (0.01) = slow learning. Too high (0.9) = unstable because updates are too aggressive.
- **γ (discount factor)**: 0.95-0.99 is ideal. Low γ makes the agent "short-sighted" (doesn't plan ahead). γ=1.0 can cause instability in continuing tasks.
- **ε-decay**: 0.9995 balances exploration and exploitation well. 0.999 stops exploring too early. 1.0 (no decay) wastes time exploring forever.

---

## 🔴 Solution 3.6: UCB Exploration (Partial — for guidance)

```python
"""
Solution 3.6: UCB Exploration in Q-Learning
────────────────────────────────────────────
Replaces ε-greedy with Upper Confidence Bound for action selection.
"""
import numpy as np

def q_learning_ucb(env, n_episodes=10000, alpha=0.1, gamma=0.99, c=2.0):
    """Q-Learning with UCB exploration instead of ε-greedy."""
    Q = np.zeros((env.n_states, env.n_actions))
    N_state = np.zeros(env.n_states)           # State visit counts
    N_sa = np.zeros((env.n_states, env.n_actions))  # State-action counts
    total_steps = 0
    rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            N_state[state] += 1

            # UCB action selection
            unvisited = np.where(N_sa[state] == 0)[0]
            if len(unvisited) > 0:
                # Force exploration of unvisited actions
                action = np.random.choice(unvisited)
            else:
                # UCB formula: Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
                ucb_values = Q[state] + c * np.sqrt(
                    np.log(N_state[state]) / N_sa[state]
                )
                action = np.argmax(ucb_values)

            next_state, reward, done = env.step(action)

            N_sa[state, action] += 1
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + gamma * best_next * (1 - done) - Q[state, action]
            )

            state = next_state
            total_reward += reward
            total_steps += 1

        rewards.append(total_reward)

    return Q, rewards, N_sa

# Train and compare
# env = GridWorld()  # From 01_q_learning_gridworld.py
# Q_ucb, rewards_ucb, visit_counts = q_learning_ucb(env, c=2.0)
# Q_eps, rewards_eps, _ = q_learning(env)  # Standard ε-greedy

# Compare:
# 1. Plot learning curves
# 2. Compare final success rates
# 3. Visualize N_sa as a heatmap to see exploration patterns
```

**Key Insight**: UCB explores more *systematically* than ε-greedy. Instead of randomly trying actions, it specifically targets actions with high uncertainty (low visit counts). This often leads to faster convergence because no pulls are "wasted" on well-understood bad actions.

---

## 📌 Note to Students

Solutions are provided for learning purposes. The goal is not to copy them, but to:
1. **Try the exercise first** (spend at least 15-30 minutes)
2. **Compare your approach** with the solution
3. **Understand the reasoning** behind design choices
4. **Improve your solution** based on what you learned

> "Understanding RL requires experimenting with it. Change the parameters, break the code, see what happens, and then fix it." — Adapted from Sutton & Barto
