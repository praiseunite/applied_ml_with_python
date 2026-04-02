# Session 03 — Exercises

## 🟢 Beginner Exercises

### Exercise 3.1: Implement ε-Greedy Bandit

**Objective**: Build your first RL agent — an ε-greedy bandit solver.

**Instructions**:
1. Create a simple 5-armed bandit with reward probabilities `[0.1, 0.3, 0.7, 0.4, 0.2]`
2. Implement the ε-greedy strategy with `ε = 0.1`
3. Run for 1000 steps
4. Plot the estimated arm values vs. true values
5. Print the percentage of times the best arm was selected

**Starter Code**:
```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleBandit:
    """A 5-armed bandit with known reward probabilities."""
    def __init__(self):
        self.true_probs = [0.1, 0.3, 0.7, 0.4, 0.2]
        self.n_arms = 5
        self.best_arm = 2  # Arm with probability 0.7

    def pull(self, arm):
        """Returns 1 (reward) with probability true_probs[arm]."""
        return float(np.random.random() < self.true_probs[arm])

# ε-Greedy agent
bandit = SimpleBandit()
n_steps = 1000
epsilon = 0.1

# Track estimates and counts
Q = np.zeros(bandit.n_arms)     # Estimated value of each arm
N = np.zeros(bandit.n_arms)     # Number of times each arm was pulled
optimal_count = 0               # Times we picked the best arm

for t in range(n_steps):
    # TODO: Implement ε-greedy action selection
    # Hint: With probability ε, choose random arm
    #       With probability 1-ε, choose arm with highest Q
    action = # YOUR CODE HERE

    # Get reward
    reward = bandit.pull(action)

    # TODO: Update Q and N using incremental mean
    # Hint: Q[a] ← Q[a] + (1/N[a]) * (reward - Q[a])
    # YOUR CODE HERE

    if action == bandit.best_arm:
        optimal_count += 1

# Print results
print(f"Optimal arm selected: {optimal_count / n_steps:.1%}")
print(f"\nEstimated vs True values:")
for i in range(bandit.n_arms):
    print(f"  Arm {i}: Q̂ = {Q[i]:.3f}  (true = {bandit.true_probs[i]:.1f})")

# TODO: Create a bar chart comparing estimated vs true values
```

**Expected Output**: 
- The best arm (Arm 2) should be selected ~90% of the time
- Estimated Q-values should be close to true probabilities

---

### Exercise 3.2: Trace Q-Learning by Hand

**Objective**: Understand Q-Learning updates by computing them manually.

**Instructions**:
Consider this tiny 3-state MDP:

```
States: {S0, S1, S2}
Actions: {LEFT, RIGHT}
S0 is the start state
S2 is the goal state (+10 reward)

Transitions:
  S0 + RIGHT → S1 (reward = -1)
  S1 + RIGHT → S2 (reward = +10, terminal)
  S1 + LEFT  → S0 (reward = -1)
  S0 + LEFT  → S0 (reward = -1, stays in place)
```

Initialize Q-table to all zeros. Use `α = 0.5`, `γ = 0.9`.

**Part A**: Calculate these Q-learning updates by hand:

| Step | State | Action | Reward | Next State | Q-Update Calculation | New Q(s,a) |
|------|-------|--------|--------|------------|---------------------|------------|
| 1 | S0 | RIGHT | -1 | S1 | Q(S0,R) ← 0 + 0.5×[-1 + 0.9×max(0,0) - 0] | ? |
| 2 | S1 | RIGHT | +10 | S2 (done) | Q(S1,R) ← 0 + 0.5×[10 + 0.9×0 - 0] | ? |
| 3 | S0 | RIGHT | -1 | S1 | Q(S0,R) ← ? + 0.5×[-1 + 0.9×max(?,?) - ?] | ? |
| 4 | S1 | LEFT | -1 | S0 | Q(S1,L) ← 0 + 0.5×[-1 + 0.9×max(?,0) - 0] | ? |
| 5 | S0 | RIGHT | -1 | S1 | Q(S0,R) ← ? + 0.5×[-1 + 0.9×max(?,?) - ?] | ? |

**Part B**: After these 5 steps, write out the complete Q-table.

**Part C**: What policy does this Q-table suggest? Is it optimal? Why or why not?

**Part D**: Verify your answers by implementing the updates in Python:
```python
import numpy as np

Q = np.zeros((3, 2))  # 3 states, 2 actions (LEFT=0, RIGHT=1)
alpha = 0.5
gamma = 0.9

# Step 1: S0, RIGHT, reward=-1, next=S1
Q[0, 1] = Q[0, 1] + alpha * (-1 + gamma * np.max(Q[1]) - Q[0, 1])
print(f"Step 1: Q(S0, RIGHT) = {Q[0, 1]:.4f}")

# TODO: Complete steps 2-5
```

---

### Exercise 3.3: Explore the GridWorld

**Objective**: Modify the GridWorld environment and observe how Q-Learning adapts.

**Instructions**:
1. Using `01_q_learning_gridworld.py` as a starting point
2. Modify the GridWorld to add a **second goal** at position 12 with reward +0.5
3. Run Q-Learning and observe: does the agent go to the +0.5 or +1.0 goal?
4. Now change the step penalty from -0.04 to -0.2. Run again. What changes?
5. Write a paragraph explaining how the step penalty affects the learned policy.

---

## 🟡 Intermediate Exercises

### Exercise 3.4: Q-Learning vs SARSA Comparison on FrozenLake

**Objective**: Compare Q-Learning and SARSA on a stochastic environment.

**Instructions**:
1. Implement the `FrozenLake` environment from the README (Part 3)
2. Train both Q-Learning and SARSA agents for 10,000 episodes
3. Test each trained agent for 1,000 episodes (greedy, no exploration)
4. Create a comparison table:

| Metric | Q-Learning | SARSA |
|--------|------------|-------|
| Training success rate (last 1000) | | |
| Test success rate (1000 episodes) | | |
| Average steps to goal (test) | | |
| Average reward (last 1000 train) | | |

5. Plot learning curves (reward per episode, smoothed) for both
6. Explain in 100+ words: Why do they differ?

**Starter Code**:
```python
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
        if action == 0: col = max(0, col - 1)        # LEFT
        elif action == 1: row = min(3, row + 1)       # DOWN
        elif action == 2: col = min(3, col + 1)       # RIGHT
        elif action == 3: row = max(0, row - 1)       # UP
        self.state = row * self.grid_size + col

        if self.state in self.holes:
            return self.state, -1.0, True
        elif self.state == self.goal:
            return self.state, 1.0, True
        else:
            return self.state, -0.01, False

# TODO: Implement q_learning() function
# TODO: Implement sarsa() function
# TODO: Train both, test both, compare results
```

---

### Exercise 3.5: Hyperparameter Sensitivity Analysis

**Objective**: Systematically explore how α, γ, and ε affect Q-Learning.

**Instructions**:
1. Using the FrozenLake environment
2. Run Q-Learning with these parameter grids (one parameter at a time):

| Parameter | Values to Test |
|-----------|---------------|
| α (learning rate) | 0.01, 0.05, 0.1, 0.3, 0.5, 0.9 |
| γ (discount factor) | 0.5, 0.8, 0.9, 0.95, 0.99, 1.0 |
| ε-decay | 0.999, 0.9995, 0.9999, 1.0 (constant) |

3. For each experiment, train for 5000 episodes and measure:
   - Final success rate (test over 500 episodes)
   - Convergence speed (episode where success rate first exceeds 80%)
   - Final average reward

4. Create 3 plots (one for each parameter), showing success rate vs. parameter value
5. Write recommendations: what are the best values and why?

**Hint**: Keep other parameters fixed at their defaults while varying one:
- Default: `α=0.1, γ=0.99, ε_start=1.0, ε_decay=0.9995, ε_min=0.01`

---

## 🔴 Advanced Exercises

### Exercise 3.6: Implement UCB Exploration Strategy

**Objective**: Replace ε-greedy with Upper Confidence Bound (UCB) exploration in Q-Learning.

**Instructions**:
1. Start with Q-Learning on GridWorld from `01_q_learning_gridworld.py`
2. Replace the ε-greedy action selection with UCB:
   
   ```
   UCB_score(s, a) = Q(s, a) + c * √(ln(N(s)) / N(s, a))
   ```
   
   Where:
   - `N(s)` = number of times state s was visited
   - `N(s, a)` = number of times action a was taken in state s
   - `c` = exploration constant (try c = 1.0, 2.0, 5.0)

3. Handle the case where `N(s, a) = 0` (force exploration of unvisited actions)
4. Compare UCB Q-Learning vs ε-Greedy Q-Learning:
   - Training convergence speed
   - Final policy quality (success rate)
   - Exploration pattern (how many unique state-action pairs explored)
5. Visualize: plot visited state-action counts as a heatmap

**Expected**: UCB should explore more systematically than ε-greedy and converge faster.

---

### Exercise 3.7: Policy Gradient on Custom Environment

**Objective**: Implement REINFORCE on a custom environment with continuous features.

**Instructions**:
1. Create a "Windy Corridor" environment:
   - Agent starts at position 0, goal at position 10
   - Actions: move left (-1) or right (+1)
   - Wind: at each step, the agent is blown left by `wind_strength` (random 0-2)
   - Reward: +10 at goal, -1 per step, -5 if blown past position -5
   - State: current position (continuous, not discrete!)

2. Implement the REINFORCE (Policy Gradient) algorithm:
   - Use the softmax policy from the README (Part 5)
   - Discretize the position into 20 bins for the state representation
   - Train for 2000 episodes

3. Compare with Q-Learning on the same environment:
   - Which converges faster?
   - Which handles the stochasticity better?

4. Plot:
   - Learning curves for both methods
   - The Policy Gradient agent's action probabilities at each position
   - A histogram of episode lengths over training

5. Write a 150+ word analysis: When would Policy Gradient outperform Q-Learning?

---

## 📝 Submission Guidelines

- Submit all code as `.py` files or Jupyter notebooks (`.ipynb`)
- Include comments explaining your reasoning
- All code must run without errors
- Include visualizations where requested
- Written analyses should be thoughtful and specific (not generic)
- Name your files: `exercise_3_X.py` (where X is the exercise number)
