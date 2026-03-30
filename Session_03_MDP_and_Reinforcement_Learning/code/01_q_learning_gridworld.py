"""
============================================================
01_q_learning_gridworld.py
Applied Machine Learning Using Python — Session 03
============================================================
Topic: Q-Learning on Custom GridWorld
Level: 🟢 Beginner → 🟡 Intermediate

Implements:
1. Custom GridWorld environment (FrozenLake-like)
2. Q-Learning algorithm from scratch
3. Training with epsilon-greedy exploration
4. Visualization of learning progress and learned policy

Usage:
    python 01_q_learning_gridworld.py
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def print_header(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


class GridWorld:
    """
    A 4x4 GridWorld environment.

    Layout:
        S  .  .  .
        .  X  .  T
        .  .  .  .
        .  X  .  G

    S = Start (0,0)
    G = Goal (+1 reward)
    T = Trap (-1 reward)
    X = Wall (impassable)
    . = Empty (-0.04 step penalty)
    """

    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
        self.action_names = ["←", "↓", "→", "↑"]

        # Special cells
        self.start = 0
        self.goal = 15
        self.traps = {7}
        self.walls = {5, 13}

        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def _pos_to_state(self, row, col):
        return row * self.grid_size + col

    def _state_to_pos(self, state):
        return divmod(state, self.grid_size)

    def step(self, action):
        row, col = self._state_to_pos(self.state)

        # Calculate new position
        if action == 0:    new_row, new_col = row, max(0, col - 1)           # LEFT
        elif action == 1:  new_row, new_col = min(3, row + 1), col           # DOWN
        elif action == 2:  new_row, new_col = row, min(3, col + 1)           # RIGHT
        elif action == 3:  new_row, new_col = max(0, row - 1), col           # UP
        else:
            raise ValueError(f"Invalid action: {action}")

        new_state = self._pos_to_state(new_row, new_col)

        # Check walls — stay in place if hitting a wall
        if new_state in self.walls:
            new_state = self.state

        self.state = new_state

        # Check outcomes
        if self.state == self.goal:
            return self.state, 1.0, True
        elif self.state in self.traps:
            return self.state, -1.0, True
        else:
            return self.state, -0.04, False

    def render(self, Q=None, agent_pos=None):
        """Print the grid with optional Q-values or agent position."""
        symbols = {self.start: "S", self.goal: "★"}
        for t in self.traps:
            symbols[t] = "T"
        for w in self.walls:
            symbols[w] = "■"

        print("┌────┬────┬────┬────┐")
        for row in range(self.grid_size):
            line = "│"
            for col in range(self.grid_size):
                s = self._pos_to_state(row, col)
                if agent_pos == s:
                    line += " 🤖 │"
                elif s in symbols:
                    line += f" {symbols[s]:>2s} │"
                elif Q is not None and s not in self.walls:
                    best_action = np.argmax(Q[s])
                    line += f" {self.action_names[best_action]:>2s} │"
                else:
                    line += "  . │"
            print(line)
            if row < self.grid_size - 1:
                print("├────┼────┼────┼────┤")
        print("└────┴────┴────┴────┘")


def q_learning(env, n_episodes=5000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
    """
    Q-Learning algorithm.

    Parameters
    ----------
    env : GridWorld environment
    n_episodes : Number of training episodes
    alpha : Learning rate
    gamma : Discount factor
    epsilon_start : Initial exploration rate
    epsilon_end : Minimum exploration rate
    epsilon_decay : Decay multiplier per episode
    """
    Q = np.zeros((env.n_states, env.n_actions))
    epsilon = epsilon_start
    rewards_history = []
    steps_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 100:  # Max steps per episode
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)

            # Q-Learning update (off-policy)
            best_next_q = np.max(Q[next_state])
            td_target = reward + gamma * best_next_q * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        rewards_history.append(total_reward)
        steps_history.append(steps)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q, rewards_history, steps_history


def main():
    # ══════════════════════════════════════════
    # STEP 1: Create Environment
    # ══════════════════════════════════════════
    print_header("STEP 1: GridWorld Environment")

    env = GridWorld()
    print("\n  Grid Layout:")
    print("  S=Start, ★=Goal, T=Trap, ■=Wall")
    env.render()

    # ══════════════════════════════════════════
    # STEP 2: Train Q-Learning Agent
    # ══════════════════════════════════════════
    print_header("STEP 2: Training Q-Learning Agent")

    Q, rewards, steps = q_learning(env, n_episodes=10000)

    # Training stats at intervals
    for i in [0, 100, 500, 1000, 5000, 9999]:
        window = rewards[max(0, i - 99):i + 1]
        avg = np.mean(window)
        print(f"  Episode {i + 1:>5d}: Avg Reward (100) = {avg:>+6.3f}")

    # ══════════════════════════════════════════
    # STEP 3: Evaluate Learned Policy
    # ══════════════════════════════════════════
    print_header("STEP 3: Learned Policy")

    print("\n  Best action in each state:")
    env.render(Q=Q)

    # Test the learned policy
    successes = 0
    n_test = 1000
    for _ in range(n_test):
        state = env.reset()
        done = False
        steps_taken = 0
        while not done and steps_taken < 50:
            action = np.argmax(Q[state])
            state, reward, done = env.step(action)
            steps_taken += 1
        if reward > 0:
            successes += 1

    print(f"\n  Test Results ({n_test} episodes):")
    print(f"  Success Rate: {successes / n_test:.1%}")

    # ══════════════════════════════════════════
    # STEP 4: Q-Table Analysis
    # ══════════════════════════════════════════
    print_header("STEP 4: Q-Table Analysis")

    print(f"\n  Q-values for key states:")
    print(f"  {'State':>8s} {'LEFT':>8s} {'DOWN':>8s} {'RIGHT':>8s} {'UP':>8s} {'Best':>6s}")
    print(f"  {'─' * 50}")

    key_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 14]
    for s in key_states:
        if s in env.walls:
            continue
        best = env.action_names[np.argmax(Q[s])]
        print(f"  {s:>8d}", end="")
        for a in range(4):
            print(f" {Q[s, a]:>8.3f}", end="")
        print(f" {best:>6s}")

    # ══════════════════════════════════════════
    # STEP 5: Visualizations
    # ══════════════════════════════════════════
    print_header("STEP 5: Training Visualizations")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Q-Learning: GridWorld Training Results",
                 fontsize=16, fontweight="bold")

    # Reward history (smoothed)
    window = 100
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(smoothed, linewidth=1, color="#3498db")
    axes[0].set_title("Average Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward (100-ep rolling avg)")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Steps per episode (smoothed)
    smoothed_steps = np.convolve(steps, np.ones(window) / window, mode="valid")
    axes[1].plot(smoothed_steps, linewidth=1, color="#2ecc71")
    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps (100-ep rolling avg)")

    # Q-value heatmap for best actions
    q_best = np.max(Q, axis=1).reshape(4, 4)
    for w in env.walls:
        r, c = divmod(w, 4)
        q_best[r, c] = np.nan
    im = axes[2].imshow(q_best, cmap="RdYlGn", interpolation="nearest")
    axes[2].set_title("State Values (max Q)")
    for i in range(4):
        for j in range(4):
            s = i * 4 + j
            if s in env.walls:
                axes[2].text(j, i, "■", ha="center", va="center", fontsize=14)
            elif s == env.goal:
                axes[2].text(j, i, "★", ha="center", va="center", fontsize=14)
            elif s in env.traps:
                axes[2].text(j, i, "T", ha="center", va="center", fontsize=14)
            else:
                axes[2].text(j, i, env.action_names[np.argmax(Q[s])],
                             ha="center", va="center", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig("../notebooks/q_learning_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: q_learning_results.png")
    plt.show()

    print("\n✅ Q-Learning Complete!")
    print("   The agent learned to navigate from S to ★ while avoiding traps!")


if __name__ == "__main__":
    main()
