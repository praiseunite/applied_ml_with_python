"""
============================================================
02_sarsa_comparison.py
Applied Machine Learning Using Python — Session 03
============================================================
Topic: SARSA vs Q-Learning — On-Policy vs Off-Policy Comparison
Level: 🟡 Intermediate → 🔴 Advanced

Demonstrates:
1. SARSA (State-Action-Reward-State-Action) implementation
2. Head-to-head comparison with Q-Learning
3. Cliff-Walking environment (classic RL benchmark)
4. Safety-conscious vs optimal-path behavior analysis
5. Hyperparameter sensitivity analysis

Why Cliff-Walking?
    This environment perfectly shows the difference between
    Q-Learning (off-policy, risk-taking) and SARSA (on-policy,
    risk-averse). SARSA learns a safer path that avoids the
    cliff edge, while Q-Learning learns the optimal but
    dangerous path right along the cliff.

Usage:
    python 02_sarsa_comparison.py
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.style.use("seaborn-v0_8-whitegrid")


def print_header(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ══════════════════════════════════════════════════════════
# ENVIRONMENT: Cliff Walking
# ══════════════════════════════════════════════════════════

class CliffWalking:
    """
    The Cliff Walking Problem (Sutton & Barto, Example 6.6).

    Grid Layout (4 rows × 12 cols):
        ┌──────────────────────────────────────────────┐
        │  .   .   .   .   .   .   .   .   .   .   .  .│  Row 0
        │  .   .   .   .   .   .   .   .   .   .   .  .│  Row 1
        │  .   .   .   .   .   .   .   .   .   .   .  .│  Row 2
        │  S  ☠️  ☠️  ☠️  ☠️  ☠️  ☠️  ☠️  ☠️  ☠️  ☠️  G│  Row 3
        └──────────────────────────────────────────────┘

    S = Start (3, 0)
    G = Goal (3, 11)
    ☠️ = Cliff cells (3, 1) through (3, 10) → reward = -100
    All other steps → reward = -1

    The agent must travel from S to G. Walking off the cliff
    sends the agent back to S with a -100 penalty.
    """

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.n_states = self.rows * self.cols
        self.n_actions = 4  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_names = ["↑", "→", "↓", "←"]

        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}

        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state

        # Movement
        if action == 0:   row = max(0, row - 1)               # UP
        elif action == 1: col = min(self.cols - 1, col + 1)    # RIGHT
        elif action == 2: row = min(self.rows - 1, row + 1)    # DOWN
        elif action == 3: col = max(0, col - 1)                # LEFT

        new_state = (row, col)

        # Check cliff → back to start with heavy penalty
        if new_state in self.cliff:
            self.state = self.start
            return self.state, -100, False

        # Check goal → terminal state
        if new_state == self.goal:
            self.state = new_state
            return self.state, -1, True

        # Normal step
        self.state = new_state
        return self.state, -1, False

    def render_policy(self, Q, title="Policy"):
        """Render the learned policy as a grid of arrows."""
        print(f"\n  {title}:")
        print("  ┌" + "────┬" * 11 + "────┐")
        for row in range(self.rows):
            line = "  │"
            for col in range(self.cols):
                state = (row, col)
                if state == self.start:
                    line += " S  │"
                elif state == self.goal:
                    line += " G  │"
                elif state in self.cliff:
                    line += " ☠  │"
                elif state in Q and len(Q[state]) > 0:
                    best = max(Q[state], key=Q[state].get)
                    line += f" {self.action_names[best]}  │"
                else:
                    line += " .  │"
            print(line)
            if row < self.rows - 1:
                print("  ├" + "────┼" * 11 + "────┤")
        print("  └" + "────┴" * 11 + "────┘")


# ══════════════════════════════════════════════════════════
# HELPER: Epsilon-Greedy Action Selection
# ══════════════════════════════════════════════════════════

def epsilon_greedy(Q, state, n_actions, epsilon):
    """Choose action using ε-greedy strategy."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        q_values = [Q[state].get(a, 0.0) for a in range(n_actions)]
        return int(np.argmax(q_values))


# ══════════════════════════════════════════════════════════
# ALGORITHM 1: Q-Learning (Off-Policy)
# ══════════════════════════════════════════════════════════

def q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0,
               epsilon=0.1):
    """
    Q-Learning: Off-policy TD control.

    Update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
                                       ^^^^^^^^^^^^^^^^^^^
                                       Uses BEST next action
                                       (regardless of what agent
                                       actually does next)

    Parameters
    ----------
    env : CliffWalking environment
    n_episodes : Number of training episodes
    alpha : Learning rate
    gamma : Discount factor (1.0 = no discounting for this task)
    epsilon : Exploration rate (kept constant for fair comparison)
    """
    Q = defaultdict(lambda: defaultdict(float))
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            action = epsilon_greedy(Q, state, env.n_actions, epsilon)
            next_state, reward, done = env.step(action)

            # Q-Learning update: uses max over next actions
            best_next_q = max(
                [Q[next_state].get(a, 0.0) for a in range(env.n_actions)]
            )
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


# ══════════════════════════════════════════════════════════
# ALGORITHM 2: SARSA (On-Policy)
# ══════════════════════════════════════════════════════════

def sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0,
          epsilon=0.1):
    """
    SARSA: On-policy TD control.

    Update: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
                                       ^^^^^^^^^
                                       Uses ACTUAL next action
                                       (the one the agent will
                                       really take next)

    Key difference from Q-Learning:
        SARSA learns the value of the policy it actually follows
        (including exploration), making it more conservative.
        Q-Learning learns the value of the optimal policy,
        making it more aggressive.

    Parameters
    ----------
    env : CliffWalking environment
    n_episodes : Number of training episodes
    alpha : Learning rate
    gamma : Discount factor
    epsilon : Exploration rate
    """
    Q = defaultdict(lambda: defaultdict(float))
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env.n_actions, epsilon)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            next_state, reward, done = env.step(action)

            # Choose next action (this IS the action agent will take)
            next_action = epsilon_greedy(
                Q, next_state, env.n_actions, epsilon
            )

            # SARSA update: uses actual next action
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


# ══════════════════════════════════════════════════════════
# ALGORITHM 3: Expected SARSA (Bridge between both)
# ══════════════════════════════════════════════════════════

def expected_sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0,
                   epsilon=0.1):
    """
    Expected SARSA uses the EXPECTED value over all possible
    next actions (weighted by the policy probability), instead
    of a single sample (SARSA) or the max (Q-Learning).

    Update: Q(s,a) ← Q(s,a) + α[r + γ·E[Q(s',a')] - Q(s,a)]

    This combines the stability of SARSA with the efficiency
    of Q-Learning.
    """
    Q = defaultdict(lambda: defaultdict(float))
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            action = epsilon_greedy(Q, state, env.n_actions, epsilon)
            next_state, reward, done = env.step(action)

            # Expected value under ε-greedy policy
            q_values = [Q[next_state].get(a, 0.0)
                        for a in range(env.n_actions)]
            best_a = int(np.argmax(q_values))

            expected_q = 0.0
            for a in range(env.n_actions):
                if a == best_a:
                    prob = (1 - epsilon) + epsilon / env.n_actions
                else:
                    prob = epsilon / env.n_actions
                expected_q += prob * q_values[a]

            # Update with expected Q-value
            td_target = reward + gamma * expected_q
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


# ══════════════════════════════════════════════════════════
# VISUALIZATION: Extract Path from Policy
# ══════════════════════════════════════════════════════════

def extract_path(env, Q, max_steps=50):
    """Follow the greedy policy to extract the learned path."""
    state = env.reset()
    path = [state]
    done = False
    steps = 0

    while not done and steps < max_steps:
        q_values = [Q[state].get(a, 0.0) for a in range(env.n_actions)]
        action = int(np.argmax(q_values))
        state, _, done = env.step(action)
        path.append(state)
        steps += 1

    return path


def main():
    # ══════════════════════════════════════════════════════
    # STEP 1: The Cliff Walking Environment
    # ══════════════════════════════════════════════════════
    print_header("STEP 1: Cliff Walking Environment")

    env = CliffWalking()
    print("""
  The Cliff Walking Problem (Sutton & Barto, Example 6.6)

  ┌────────────────────────────────────────────────────────┐
  │  .    .    .    .    .    .    .    .    .    .    .   .│  Row 0
  │  .    .    .    .    .    .    .    .    .    .    .   .│  Row 1
  │  .    .    .    .    .    .    .    .    .    .    .   .│  Row 2
  │  S    ☠    ☠    ☠    ☠    ☠    ☠    ☠    ☠    ☠    ☠   G│  Row 3
  └────────────────────────────────────────────────────────┘

  S = Start       G = Goal       ☠ = Cliff (-100 penalty)
  Each step costs -1. The optimal path has total reward = -13.
    """)

    # ══════════════════════════════════════════════════════
    # STEP 2: Train All Three Algorithms
    # ══════════════════════════════════════════════════════
    print_header("STEP 2: Training Three Algorithms (500 episodes each)")

    n_episodes = 500
    n_runs = 10  # Average over multiple runs for stability

    all_rewards = {"Q-Learning": [], "SARSA": [], "Expected SARSA": []}

    for run in range(n_runs):
        env_q = CliffWalking()
        env_s = CliffWalking()
        env_e = CliffWalking()

        _, r_q = q_learning(env_q, n_episodes=n_episodes)
        _, r_s = sarsa(env_s, n_episodes=n_episodes)
        _, r_e = expected_sarsa(env_e, n_episodes=n_episodes)

        all_rewards["Q-Learning"].append(r_q)
        all_rewards["SARSA"].append(r_s)
        all_rewards["Expected SARSA"].append(r_e)

        if (run + 1) % 5 == 0:
            print(f"  Completed run {run + 1}/{n_runs}")

    # Average across runs
    avg_rewards = {}
    for name, runs in all_rewards.items():
        avg_rewards[name] = np.mean(runs, axis=0)

    # ══════════════════════════════════════════════════════
    # STEP 3: Compare Learned Policies
    # ══════════════════════════════════════════════════════
    print_header("STEP 3: Learned Policies Comparison")

    # Train single final models for policy display
    env_q = CliffWalking()
    env_s = CliffWalking()
    env_e = CliffWalking()

    Q_q, _ = q_learning(env_q, n_episodes=1000)
    Q_s, _ = sarsa(env_s, n_episodes=1000)
    Q_e, _ = expected_sarsa(env_e, n_episodes=1000)

    env_q.render_policy(Q_q, "Q-Learning Policy (Optimal but Risky)")
    env_s.render_policy(Q_s, "SARSA Policy (Safe Path)")
    env_e.render_policy(Q_e, "Expected SARSA Policy")

    # Extract and display paths
    path_q = extract_path(env_q, Q_q)
    path_s = extract_path(env_s, Q_s)
    path_e = extract_path(env_e, Q_e)

    print(f"\n  Q-Learning path length:       {len(path_q)} steps")
    print(f"  SARSA path length:            {len(path_s)} steps")
    print(f"  Expected SARSA path length:   {len(path_e)} steps")

    # ══════════════════════════════════════════════════════
    # STEP 4: Performance Analysis
    # ══════════════════════════════════════════════════════
    print_header("STEP 4: Performance Analysis")

    window = 20
    for name, rewards in avg_rewards.items():
        final_avg = np.mean(rewards[-50:])
        best = np.max(np.convolve(rewards, np.ones(window)/window, "valid"))
        cliffs = sum(1 for r_list in all_rewards[name]
                     for r in r_list if r < -50)
        print(f"\n  {name}:")
        print(f"    Final avg reward (last 50): {final_avg:>8.1f}")
        print(f"    Best rolling avg ({window}-ep):    {best:>8.1f}")
        print(f"    Total cliff falls:          {cliffs:>8d}")

    # ══════════════════════════════════════════════════════
    # STEP 5: Visualizations
    # ══════════════════════════════════════════════════════
    print_header("STEP 5: Generating Visualizations")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("SARSA vs Q-Learning: Cliff Walking Comparison",
                 fontsize=18, fontweight="bold", y=0.98)

    colors = {
        "Q-Learning": "#e74c3c",
        "SARSA": "#2ecc71",
        "Expected SARSA": "#3498db"
    }

    # ─── Plot 1: Smoothed Reward Curves ───
    ax = axes[0, 0]
    window = 20
    for name, rewards in avg_rewards.items():
        smoothed = np.convolve(rewards, np.ones(window)/window, "valid")
        ax.plot(smoothed, label=name, color=colors[name],
                linewidth=2, alpha=0.9)
    ax.set_title("Reward per Episode (Smoothed)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sum of Rewards")
    ax.axhline(y=-13, color="gray", linestyle="--", alpha=0.5,
               label="Optimal (-13)")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(-150, 0)
    ax.grid(True, alpha=0.3)

    # ─── Plot 2: Cumulative Reward ───
    ax = axes[0, 1]
    for name, rewards in avg_rewards.items():
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, label=name, color=colors[name],
                linewidth=2, alpha=0.9)
    ax.set_title("Cumulative Reward", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Cumulative Reward")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ─── Plot 3: Path Visualization ───
    ax = axes[1, 0]
    # Draw grid
    for i in range(5):
        ax.axhline(y=i, color="gray", linewidth=0.5, alpha=0.3)
    for j in range(13):
        ax.axvline(x=j, color="gray", linewidth=0.5, alpha=0.3)

    # Draw cliff
    for c in range(1, 11):
        ax.add_patch(plt.Rectangle((c, 0), 1, 1, color="#e74c3c",
                                    alpha=0.3))
        ax.text(c + 0.5, 0.5, "☠", ha="center", va="center", fontsize=10)

    # Draw start and goal
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#2ecc71", alpha=0.4))
    ax.text(0.5, 0.5, "S", ha="center", va="center", fontsize=14,
            fontweight="bold")
    ax.add_patch(plt.Rectangle((11, 0), 1, 1, color="#f1c40f", alpha=0.4))
    ax.text(11.5, 0.5, "G", ha="center", va="center", fontsize=14,
            fontweight="bold")

    # Draw paths
    def draw_path(path, color, label, offset=0.0):
        rows = [4 - 1 - r + offset for r, c in path]
        cols = [c + 0.5 + offset for r, c in path]
        ax.plot(cols, rows, color=color, linewidth=3, alpha=0.8,
                label=label, marker=".", markersize=6)

    draw_path(path_q, colors["Q-Learning"], "Q-Learning", offset=0.0)
    draw_path(path_s, colors["SARSA"], "SARSA", offset=0.05)
    draw_path(path_e, colors["Expected SARSA"], "Expected SARSA",
              offset=-0.05)

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title("Learned Paths", fontsize=13, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row (inverted)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")

    # ─── Plot 4: Algorithm Comparison Summary ───
    ax = axes[1, 1]
    algorithms = list(avg_rewards.keys())
    metrics = {
        "Final Reward\n(last 50 avg)": [],
        "Cliff Falls\n(per run avg)": [],
        "Convergence\n(episodes to -20)": [],
    }

    for name in algorithms:
        rewards = avg_rewards[name]
        metrics["Final Reward\n(last 50 avg)"].append(
            -np.mean(rewards[-50:])
        )

        cliff_count = np.mean([
            sum(1 for r in run if r < -50)
            for run in all_rewards[name]
        ])
        metrics["Cliff Falls\n(per run avg)"].append(cliff_count)

        # Convergence: first episode where rolling avg > -20
        smoothed = np.convolve(rewards, np.ones(20)/20, "valid")
        converged = np.where(smoothed > -30)[0]
        conv_ep = converged[0] if len(converged) > 0 else n_episodes
        metrics["Convergence\n(episodes to -20)"].append(conv_ep)

    x = np.arange(len(algorithms))
    width = 0.25
    for i, (metric, values) in enumerate(metrics.items()):
        normalized = np.array(values) / max(max(values), 1)
        bars = ax.bar(x + i * width, normalized, width, label=metric,
                      alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(algorithms, fontsize=10)
    ax.set_title("Algorithm Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Score")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("../notebooks/sarsa_comparison_results.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: sarsa_comparison_results.png")
    plt.show()

    # ══════════════════════════════════════════════════════
    # STEP 6: Key Takeaways
    # ══════════════════════════════════════════════════════
    print_header("KEY TAKEAWAYS")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  SARSA vs Q-Learning — When to Use Which?               │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  Use Q-LEARNING when:                                   │
  │  • You want the theoretically optimal policy            │
  │  • Risk/cost of exploration is low                      │
  │  • Learning from replay buffers (experience replay)     │
  │  • Building Deep Q-Networks (DQN)                       │
  │                                                         │
  │  Use SARSA when:                                        │
  │  • Safety during training matters (robotics, trading)   │
  │  • The agent must follow its own policy                 │
  │  • Exploration has real-world consequences              │
  │  • You need conservative, risk-averse behavior          │
  │                                                         │
  │  Use EXPECTED SARSA when:                               │
  │  • You want the best of both worlds                     │
  │  • Lower variance than SARSA, less risky than Q-Learn   │
  │  • Computational cost is acceptable                     │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
    """)

    print("✅ SARSA Comparison Complete!")
    print("   On Cliff Walking, SARSA learns a SAFE path away from the")
    print("   cliff, while Q-Learning learns the OPTIMAL path along the")
    print("   cliff edge — but during training, it falls frequently.")


if __name__ == "__main__":
    main()
