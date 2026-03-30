# 📊 Portfolio Component — Session 03

## Assignment: RL Agent Visualization Dashboard

### Overview

Build an **interactive RL Agent Visualization Dashboard** that demonstrates your understanding of reinforcement learning by training, visualizing, and comparing RL algorithms in real-time. This project showcases both your RL knowledge and your ability to build engaging, data-driven applications.

---

### Deliverables

| # | Deliverable | Format | Estimated Time |
|---|------------|--------|----------------|
| 1 | Interactive Dashboard | Python (Gradio or Streamlit) | 3-4 hours |
| 2 | Analysis Report | Markdown (`.md`) | 1-2 hours |
| 3 | Trained Agent Demonstrations | PNG/GIF visualizations | Included |

---

### Dashboard Requirements

#### Core Features (Required — 70 points)

1. **Environment Visualization** (15 pts)
   - Display the GridWorld or FrozenLake environment graphically
   - Show the agent's current position, walls, traps, and goals
   - Animate the agent moving along its learned path
   - Color-code states by their value (heatmap overlay)

2. **Algorithm Selection & Training** (20 pts)
   - Let users choose between: Q-Learning, SARSA, Expected SARSA
   - Adjustable hyperparameters via sliders:
     - Learning rate (α): 0.01 - 1.0
     - Discount factor (γ): 0.0 - 1.0
     - Exploration rate (ε): 0.0 - 1.0
     - Number of episodes: 100 - 50,000
   - "Train" button that runs the algorithm and shows progress

3. **Results Visualization** (20 pts)
   - Learning curve (reward per episode, smoothed)
   - Learned policy displayed as arrows on the grid
   - Q-table heatmap
   - Success rate over time

4. **Algorithm Comparison** (15 pts)
   - Train multiple algorithms and display results side-by-side
   - Comparison table with key metrics
   - Overlaid learning curves

#### Advanced Features (Bonus — 30 points)

5. **Multi-Armed Bandit Module** (10 pts)
   - Interactive bandit simulator
   - Users can set arm probabilities or use presets
   - Compare ε-Greedy, UCB, and Thompson Sampling
   - Show cumulative regret in real-time

6. **Custom Environment Builder** (10 pts)
   - Let users create their own GridWorld by:
     - Clicking to place/remove walls
     - Setting start, goal, and trap positions
     - Adjusting rewards
   - Train an agent on the custom environment

7. **Step-by-Step Mode** (10 pts)
   - "Step through" training one episode at a time
   - Show Q-table updates after each step
   - Highlight the TD error and update calculation
   - Excellent for teaching/learning

---

### Implementation Guide

#### Recommended Tech Stack

```python
# Option 1: Gradio (simpler, deployable to HF Spaces)
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Option 2: Streamlit (more flexible UI)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
```

#### Starter Structure

```
portfolio_rl_dashboard/
├── app.py                 # Main application
├── environments/
│   ├── gridworld.py       # GridWorld environment
│   └── frozen_lake.py     # FrozenLake environment
├── agents/
│   ├── q_learning.py      # Q-Learning agent
│   ├── sarsa.py           # SARSA agent
│   └── bandits.py         # Multi-armed bandit agents
├── visualization/
│   ├── grid_renderer.py   # Grid visualization
│   └── plots.py           # Learning curves, heatmaps
├── requirements.txt
├── README.md
└── report.md              # Analysis report
```

#### Gradio Quick Start

```python
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Your environment and agent classes here...

def train_and_visualize(algorithm, alpha, gamma, epsilon, n_episodes):
    """Train an RL agent and return visualizations."""
    env = GridWorld()

    if algorithm == "Q-Learning":
        Q, rewards = q_learning(env, n_episodes=int(n_episodes),
                                alpha=alpha, gamma=gamma)
    elif algorithm == "SARSA":
        Q, rewards = sarsa(env, n_episodes=int(n_episodes),
                           alpha=alpha, gamma=gamma)

    # Create learning curve plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    window = max(10, int(n_episodes * 0.02))
    smoothed = np.convolve(rewards, np.ones(window)/window, 'valid')
    axes[0].plot(smoothed)
    axes[0].set_title('Learning Curve')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')

    # Create Q-value heatmap
    q_max = np.max(Q, axis=1).reshape(4, 4)
    axes[1].imshow(q_max, cmap='RdYlGn')
    axes[1].set_title('State Values')

    plt.tight_layout()
    return fig

# Build Gradio interface
demo = gr.Interface(
    fn=train_and_visualize,
    inputs=[
        gr.Dropdown(["Q-Learning", "SARSA"], label="Algorithm"),
        gr.Slider(0.01, 1.0, value=0.1, label="Learning Rate (α)"),
        gr.Slider(0.0, 1.0, value=0.99, label="Discount Factor (γ)"),
        gr.Slider(0.0, 1.0, value=0.1, label="Exploration (ε)"),
        gr.Slider(100, 50000, value=5000, step=100, label="Episodes"),
    ],
    outputs=gr.Plot(label="Results"),
    title="🤖 RL Agent Visualization Dashboard",
    description="Train and compare reinforcement learning agents interactively!",
)

demo.launch()
```

---

### Report Structure

```markdown
# RL Agent Visualization Dashboard — Report
## [Your Name] | [Date]

## 1. Project Overview
- What does the dashboard do?
- What algorithms are implemented?
- What makes it useful for learning RL?

## 2. Algorithm Implementations
- Brief description of each algorithm
- Key implementation decisions (data structures, optimizations)
- How hyperparameters are handled

## 3. Experiment Results
- Default GridWorld results for each algorithm
- Comparison table (success rate, convergence speed, safety)
- Learning curve analysis

## 4. Key Insights
- When does Q-Learning outperform SARSA? (and vice versa)
- How sensitive are results to hyperparameters?
- What surprised you during development?

## 5. Technical Challenges
- What was hardest to implement?
- How did you debug RL agents?
- Performance considerations

## 6. Future Improvements
- Deep Q-Networks (DQN)
- More environments
- Better visualizations
```

---

### Grading Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Environment Visualization** | 15 | Clear, intuitive grid display with agent, walls, goals |
| **Algorithm Implementation** | 20 | Correct Q-Learning and SARSA with adjustable parameters |
| **Results Visualization** | 20 | Learning curves, Q-table heatmap, policy arrows |
| **Algorithm Comparison** | 15 | Side-by-side comparison with metrics table |
| **Code Quality** | 10 | Clean, modular, well-documented code |
| **Report Quality** | 10 | Thoughtful analysis with supporting data |
| **UI/UX Design** | 5 | Intuitive interface, proper labels, responsive layout |
| **Bonus Features** | +30 | Bandits module, custom envs, step-by-step mode |
| **Total** | **100** (+30 bonus) |

---

### Deployment Guide

#### Deploy to Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload your files:
   ```bash
   # Install the HF CLI
   pip install huggingface_hub

   # Login
   huggingface-cli login

   # Clone your space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/rl-dashboard

   # Copy your files and push
   cp -r portfolio_rl_dashboard/* rl-dashboard/
   cd rl-dashboard
   git add .
   git commit -m "Deploy RL dashboard"
   git push
   ```

4. Your dashboard will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/rl-dashboard`

#### Requirements.txt for Deployment

```
gradio>=4.0
numpy>=1.24
matplotlib>=3.7
```

---

### Tips for an Outstanding Project

1. **Make it interactive**: The best dashboards let users experiment and learn through interaction. Add sliders, dropdowns, and real-time updates.

2. **Tell a story**: Guide users through the RL experience — "Here's how the agent starts confused → explores → learns → masters the environment."

3. **Show the 'aha' moment**: The most impactful visualization is the Cliff Walking comparison where SARSA learns a safe path while Q-Learning hugs the cliff edge.

4. **Polish the UI**: Add clear labels, helpful tooltips, and a clean layout. First impressions matter.

5. **Include educational content**: Add markdown explanations alongside your visualizations. Help the user understand *why* the results look the way they do.

---

> 💡 **Why This Matters**: RL is one of the most visually compelling areas of ML. An interactive RL dashboard on your portfolio shows hiring managers that you don't just understand algorithms theoretically — you can make them come alive in interactive applications. This is exactly what companies like DeepMind, OpenAI, and autonomous vehicle companies look for.
