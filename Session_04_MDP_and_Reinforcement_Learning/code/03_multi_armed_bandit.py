"""
============================================================
03_multi_armed_bandit.py
Applied Machine Learning Using Python — Session 03
============================================================
Topic: Multi-Armed Bandit Problem & Exploration Strategies
Level: 🟢 Beginner → 🔴 Advanced

Demonstrates:
1. Multi-Armed Bandit problem (the simplest RL problem)
2. Three exploration strategies:
   - ε-Greedy (simple and effective)
   - Upper Confidence Bound (UCB — principled exploration)
   - Thompson Sampling (Bayesian approach — often the best)
3. Comparison of cumulative regret
4. Real-world application: A/B testing for ad selection

Why Bandits Matter:
    The exploration-exploitation dilemma is the CORE problem
    in RL. Before tackling complex environments, understand
    this fundamental trade-off through bandits. They're also
    widely used in industry: clinical trials, ad selection,
    recommendation systems, and A/B testing.

Usage:
    python 03_multi_armed_bandit.py
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
# ENVIRONMENT: Multi-Armed Bandit
# ══════════════════════════════════════════════════════════

class MultiArmedBandit:
    """
    A k-armed bandit problem.

    Each arm has a fixed (but unknown) reward probability.
    Pulling arm i gives reward 1 with probability p_i, else 0.
    The agent must figure out which arm is best while
    maximizing total reward.

    Analogy: k slot machines in a casino. Each has a different
    (unknown) payout rate. How do you maximize your winnings?
    """

    def __init__(self, n_arms=10, seed=42):
        np.random.seed(seed)
        self.n_arms = n_arms
        # True reward probabilities (unknown to the agent)
        self.true_probs = np.random.beta(a=2, b=5, size=n_arms)
        # Make one arm clearly the best
        self.best_arm = np.argmax(self.true_probs)
        self.true_probs[self.best_arm] = max(self.true_probs) + 0.1
        self.true_probs = np.clip(self.true_probs, 0, 1)

    def pull(self, arm):
        """Pull an arm and get stochastic reward."""
        reward = np.random.random() < self.true_probs[arm]
        return float(reward)

    def optimal_reward(self):
        """The reward we'd get by always pulling the best arm."""
        return self.true_probs[self.best_arm]

    def __repr__(self):
        probs_str = ", ".join(f"{p:.3f}" for p in self.true_probs)
        return (f"MultiArmedBandit(n_arms={self.n_arms}, "
                f"probs=[{probs_str}], "
                f"best_arm={self.best_arm})")


# ══════════════════════════════════════════════════════════
# STRATEGY 1: ε-Greedy
# ══════════════════════════════════════════════════════════

class EpsilonGreedy:
    """
    ε-Greedy Strategy.

    With probability ε: explore (random arm)
    With probability 1-ε: exploit (best known arm)

    Pros: Simple, easy to implement
    Cons: Explores uniformly (doesn't focus on promising arms)
    """

    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.name = f"ε-Greedy (ε={epsilon})"

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n


class DecayingEpsilonGreedy:
    """
    ε-Greedy with decaying ε.

    Starts with high exploration, gradually shifts to exploitation.
    ε_t = min(1, C / t) where C controls the decay rate.
    """

    def __init__(self, n_arms, c=5.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
        self.name = f"Decaying ε-Greedy (c={c})"

    def select_arm(self):
        self.t += 1
        epsilon = min(1.0, self.c / self.t)
        if np.random.random() < epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


# ══════════════════════════════════════════════════════════
# STRATEGY 2: Upper Confidence Bound (UCB)
# ══════════════════════════════════════════════════════════

class UCB:
    """
    Upper Confidence Bound (UCB1) Strategy.

    Select arm that maximizes: Q(a) + c * √(ln(t) / N(a))
                               ────   ──────────────────
                               exploit    explore bonus

    The explore bonus is HIGH for arms pulled rarely (N(a) small)
    and decreases as an arm is pulled more. This ensures every
    arm gets tried, with preference for uncertain/promising ones.

    Pros: No random exploration — principled uncertainty-based
    Cons: Can over-explore in early stages
    """

    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
        self.name = f"UCB (c={c})"

    def select_arm(self):
        self.t += 1
        # Pull each arm once first
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.t) / self.counts
        )
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


# ══════════════════════════════════════════════════════════
# STRATEGY 3: Thompson Sampling (Bayesian)
# ══════════════════════════════════════════════════════════

class ThompsonSampling:
    """
    Thompson Sampling (Bayesian Approach).

    Maintains a Beta distribution for each arm's reward probability.
    Each round: sample from each arm's distribution, pick the highest.

    Beta(α, β):
    - α = successes + 1 (prior)
    - β = failures + 1 (prior)
    - Mean = α / (α + β)

    The beauty: arms with uncertain estimates get explored
    naturally because their samples have high variance.
    As we learn, the distributions narrow and we exploit.

    Pros: Often the best strategy, naturally balances explore/exploit
    Cons: Requires Bayesian reasoning, assumes Beta-distributed rewards
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Successes + prior
        self.beta = np.ones(n_arms)   # Failures + prior
        self.counts = np.zeros(n_arms)
        self.name = "Thompson Sampling"

    def select_arm(self):
        # Sample from each arm's Beta posterior
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.counts[arm] += 1
        # Update Beta distribution
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)


# ══════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════

def run_experiment(bandit, strategy, n_steps=5000):
    """Run a single bandit experiment and track metrics."""
    rewards = np.zeros(n_steps)
    regrets = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)

    for t in range(n_steps):
        arm = strategy.select_arm()
        reward = bandit.pull(arm)
        strategy.update(arm, reward)

        rewards[t] = reward
        regrets[t] = bandit.optimal_reward() - bandit.true_probs[arm]
        optimal_actions[t] = float(arm == bandit.best_arm)

    return {
        "rewards": rewards,
        "regrets": regrets,
        "cumulative_regret": np.cumsum(regrets),
        "optimal_actions": optimal_actions,
        "arm_counts": strategy.counts.copy(),
    }


def main():
    # ══════════════════════════════════════════════════════
    # STEP 1: The Multi-Armed Bandit Problem
    # ══════════════════════════════════════════════════════
    print_header("STEP 1: Multi-Armed Bandit Problem")

    bandit = MultiArmedBandit(n_arms=10, seed=42)

    print(f"\n  {bandit}")
    print(f"\n  True reward probabilities (UNKNOWN to agent):")
    for i, p in enumerate(bandit.true_probs):
        indicator = " ← BEST" if i == bandit.best_arm else ""
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"    Arm {i}: [{bar}] {p:.3f}{indicator}")

    print(f"\n  The agent doesn't know these probabilities!")
    print(f"  It must LEARN which arm is best by trying them.")

    # ══════════════════════════════════════════════════════
    # STEP 2: Run All Strategies
    # ══════════════════════════════════════════════════════
    print_header("STEP 2: Running 5 Strategies (5000 steps each)")

    n_steps = 5000
    n_runs = 30  # Average over 30 runs for stability

    strategies_config = [
        ("ε-Greedy (0.1)", lambda: EpsilonGreedy(10, epsilon=0.1)),
        ("ε-Greedy (0.01)", lambda: EpsilonGreedy(10, epsilon=0.01)),
        ("Decaying ε", lambda: DecayingEpsilonGreedy(10, c=5.0)),
        ("UCB", lambda: UCB(10, c=2.0)),
        ("Thompson", lambda: ThompsonSampling(10)),
    ]

    all_results = {}
    for name, strategy_factory in strategies_config:
        run_results = []
        for _ in range(n_runs):
            bandit = MultiArmedBandit(n_arms=10, seed=42)
            strategy = strategy_factory()
            result = run_experiment(bandit, strategy, n_steps)
            run_results.append(result)

        # Average across runs
        avg_result = {
            "rewards": np.mean([r["rewards"] for r in run_results], axis=0),
            "cumulative_regret": np.mean(
                [r["cumulative_regret"] for r in run_results], axis=0
            ),
            "optimal_actions": np.mean(
                [r["optimal_actions"] for r in run_results], axis=0
            ),
            "arm_counts": np.mean(
                [r["arm_counts"] for r in run_results], axis=0
            ),
        }
        all_results[name] = avg_result
        print(f"  ✓ {name:20s} — Final regret: "
              f"{avg_result['cumulative_regret'][-1]:.1f}")

    # ══════════════════════════════════════════════════════
    # STEP 3: Results Analysis
    # ══════════════════════════════════════════════════════
    print_header("STEP 3: Detailed Results")

    print(f"\n  {'Strategy':<22s} {'Total Regret':>14s} "
          f"{'Opt. Action %':>14s} {'Total Reward':>14s}")
    print(f"  {'─' * 66}")

    for name, res in all_results.items():
        total_regret = res["cumulative_regret"][-1]
        opt_pct = np.mean(res["optimal_actions"][-1000:]) * 100
        total_reward = np.sum(res["rewards"])
        print(f"  {name:<22s} {total_regret:>14.1f} "
              f"{opt_pct:>13.1f}% {total_reward:>14.1f}")

    # Show arm pull distribution for each strategy
    bandit = MultiArmedBandit(n_arms=10, seed=42)
    print(f"\n  Arm pull distribution (% of total pulls):")
    print(f"  {'Arm':<8s}", end="")
    for name in all_results:
        print(f" {name[:8]:>9s}", end="")
    print(f" {'True P':>9s}")
    print(f"  {'─' * 55}")

    for arm in range(10):
        indicator = " *" if arm == bandit.best_arm else ""
        print(f"  Arm {arm}{indicator:<4s}", end="")
        for name, res in all_results.items():
            pct = res["arm_counts"][arm] / n_steps * 100
            print(f" {pct:>8.1f}%", end="")
        print(f"  {bandit.true_probs[arm]:>7.3f}")

    # ══════════════════════════════════════════════════════
    # STEP 4: Visualizations
    # ══════════════════════════════════════════════════════
    print_header("STEP 4: Generating Visualizations")

    colors = {
        "ε-Greedy (0.1)": "#e74c3c",
        "ε-Greedy (0.01)": "#e67e22",
        "Decaying ε": "#9b59b6",
        "UCB": "#3498db",
        "Thompson": "#2ecc71",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Multi-Armed Bandit: Strategy Comparison",
                 fontsize=18, fontweight="bold", y=0.98)

    # ─── Plot 1: Cumulative Regret ───
    ax = axes[0, 0]
    for name, res in all_results.items():
        ax.plot(res["cumulative_regret"], label=name,
                color=colors[name], linewidth=2, alpha=0.9)
    ax.set_title("Cumulative Regret (Lower = Better)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Regret")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ─── Plot 2: Optimal Action % Over Time ───
    ax = axes[0, 1]
    window = 100
    for name, res in all_results.items():
        smoothed = np.convolve(
            res["optimal_actions"], np.ones(window)/window, "valid"
        )
        ax.plot(smoothed * 100, label=name, color=colors[name],
                linewidth=2, alpha=0.9)
    ax.set_title("% Optimal Action (Higher = Better)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("% Optimal Action (rolling avg)")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # ─── Plot 3: Average Reward Over Time ───
    ax = axes[1, 0]
    for name, res in all_results.items():
        smoothed = np.convolve(
            res["rewards"], np.ones(window)/window, "valid"
        )
        ax.plot(smoothed, label=name, color=colors[name],
                linewidth=2, alpha=0.9)
    bandit = MultiArmedBandit(n_arms=10, seed=42)
    ax.axhline(y=bandit.optimal_reward(), color="gray", linestyle="--",
               alpha=0.5, label=f"Optimal ({bandit.optimal_reward():.3f})")
    ax.set_title("Average Reward (Higher = Better)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (rolling avg)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ─── Plot 4: Arm Selection Heatmap ───
    ax = axes[1, 1]
    strategy_names = list(all_results.keys())
    arm_distributions = np.array([
        all_results[name]["arm_counts"] / n_steps * 100
        for name in strategy_names
    ])
    im = ax.imshow(arm_distributions, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"Arm {i}" for i in range(10)], fontsize=8)
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels([s[:12] for s in strategy_names], fontsize=9)
    ax.set_title("Arm Pull Distribution (%)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Arm")

    # Annotate cells
    for i in range(len(strategy_names)):
        for j in range(10):
            val = arm_distributions[i, j]
            color = "white" if val > 30 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    # Mark best arm
    ax.axvline(x=bandit.best_arm - 0.5, color="#2ecc71", linewidth=2)
    ax.axvline(x=bandit.best_arm + 0.5, color="#2ecc71", linewidth=2)

    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("../notebooks/bandit_comparison_results.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: bandit_comparison_results.png")
    plt.show()

    # ══════════════════════════════════════════════════════
    # STEP 5: Real-World Application — Ad Selection
    # ══════════════════════════════════════════════════════
    print_header("STEP 5: Real-World Application — A/B Testing")

    print("""
  Scenario: An online store has 5 ad designs. Each ad has an
  unknown click-through rate (CTR). Using Thompson Sampling,
  we can dynamically allocate traffic to better-performing ads
  while still exploring alternatives.
    """)

    class AdBandit:
        """Ads with different click-through rates."""
        def __init__(self):
            self.ctrs = [0.02, 0.05, 0.03, 0.08, 0.04]
            self.names = ["Banner A", "Banner B", "Sidebar",
                          "Pop-up", "Footer"]
            self.best_ad = np.argmax(self.ctrs)

        def show_ad(self, ad_idx):
            clicked = np.random.random() < self.ctrs[ad_idx]
            return float(clicked)

    ad_env = AdBandit()
    ts = ThompsonSampling(n_arms=5)

    impressions = np.zeros(5)
    clicks = np.zeros(5)
    n_total = 10000

    for t in range(n_total):
        ad = ts.select_arm()
        click = ad_env.show_ad(ad)
        ts.update(ad, click)
        impressions[ad] += 1
        clicks[ad] += click

    print(f"  Results after {n_total:,} impressions:\n")
    print(f"  {'Ad':<12s} {'True CTR':>10s} {'Impressions':>13s} "
          f"{'Clicks':>8s} {'Measured CTR':>14s} {'Traffic %':>10s}")
    print(f"  {'─' * 70}")

    for i in range(5):
        measured = clicks[i] / max(impressions[i], 1)
        traffic = impressions[i] / n_total * 100
        marker = " ← BEST" if i == ad_env.best_ad else ""
        print(f"  {ad_env.names[i]:<12s} {ad_env.ctrs[i]:>10.1%} "
              f"{impressions[i]:>13.0f} {clicks[i]:>8.0f} "
              f"{measured:>14.1%} {traffic:>9.1f}%{marker}")

    print(f"""
  Key Insight: Thompson Sampling automatically allocated
  ~{impressions[ad_env.best_ad]/n_total*100:.0f}% of traffic to the best ad (Pop-up)
  while still testing alternatives. No need to split traffic
  50/50 like traditional A/B testing!
    """)

    # ══════════════════════════════════════════════════════
    # STEP 6: Key Takeaways
    # ══════════════════════════════════════════════════════
    print_header("KEY TAKEAWAYS")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  Multi-Armed Bandit Strategies — Summary                │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  ε-GREEDY:                                              │
  │  • Simplest strategy — great starting point             │
  │  • Fixed exploration rate wastes pulls on bad arms      │
  │  • Decaying ε improves this significantly               │
  │                                                         │
  │  UCB (Upper Confidence Bound):                          │
  │  • Deterministic — no randomness in arm selection       │
  │  • Explores uncertain arms, not random ones             │
  │  • Strong theoretical regret guarantees                 │
  │                                                         │
  │  THOMPSON SAMPLING:                                     │
  │  • Often the best performer in practice                 │
  │  • Naturally balances explore/exploit via uncertainty    │
  │  • Used by Google, Microsoft, Netflix in production     │
  │  • Easy to extend to contextual bandits                 │
  │                                                         │
  │  INDUSTRY USAGE:                                        │
  │  • Clinical trials → adaptive dosing                    │
  │  • A/B testing → dynamic traffic allocation             │
  │  • Recommendations → explore new content                │
  │  • Ad selection → maximize click-through rate           │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
    """)

    print("✅ Multi-Armed Bandit Analysis Complete!")


if __name__ == "__main__":
    main()
