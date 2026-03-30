# 📊 Portfolio Component — Session 04

## Assignment: Real-Time Actor-Critic Training Dashboard

### Overview

In Session 3, your dashboard demonstrated how an agent behaves *after* it trains (value table visualization). In Session 4, you are working with Deep Learning and Policy Gradients, which require thousands of steps. 

For this portfolio project, you will build a **Real-Time Agent Training Monitor** using Gradio and TensorFlow. You will deploy this dashboard to Hugging Face Spaces. This shows employers you know how to build MLOps monitoring tools for live Deep Learning models.

---

### Deliverables

| # | Deliverable | Format | Estimated Time |
|---|------------|--------|----------------|
| 1 | Live Training Dashboard | Python (`app.py` via Gradio) | 3-4 hours |
| 2 | Model Weights | Saved `.h5` or `.keras` models | Included |
| 3 | Project Report | Markdown (`.md`) | 1 hour |

---

### Dashboard Requirements

#### Core Features (Required — 70 points)

1. **Algorithm Switcher** (10 pts)
   - Allow the user to select between `REINFORCE` and `Actor-Critic (A2C)`.
2. **Live Matplotlib Updates** (30 pts)
   - Do NOT wait for training to finish to show the plot.
   - The Gradio dashboard must update the training chart (Rewards vs Episodes) dynamically (e.g., updating every 10 episodes in real-time).
3. **Hyperparameter Controls** (15 pts)
   - Sliders for Actor Learning Rate (`0.0001` to `0.05`).
   - Sliders for Critic Learning Rate.
   - Sliders for Discount Factor $\gamma$ (`0.5` to `0.999`).
4. **Agent Demo Video** (15 pts)
   - Once training is complete, capture a video (or GIF) of the agent playing `CartPole-v1`.
   - Embed this video dynamically back into the Gradio UI so the user can watch the result.

#### Advanced Features (Bonus — 30 points)

5. **TensorBoard Integration** (10 pts)
   - Log the Critic Loss and Actor Gradients using `tf.summary`.
6. **Continuous Space Environment** (10 pts)
   - Add a dropdown to train on `Pendulum-v1` (requires modifying the network to output mean/std values as per Exercise 4.5).
7. **Early Stopping** (10 pts)
   - Allow the user to click a "Stop Training and Test Now" button to interrupt training safely and view the current policy results.

---

### Implementation Guide

Because you are writing a live-updating dashboard in Gradio, you must use Python `generators` (`yield`) to push updates to the UI, rather than returning the plot only at the end.

#### Starter Gradio Code with Live Updates:

```python
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import time

def train_agent_live(algorithm, actor_lr, episodes):
    # Initialize your agent based on arguments here...
    
    rewards = []
    
    for ep in range(episodes):
        # ... Run one episode ...
        fake_reward = np.random.randint(10, 50) + ep # Placeholder for actual agent.play()
        rewards.append(fake_reward)
        
        # Every 10 episodes, update the UI
        if ep % 10 == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(rewards, color='blue')
            ax.set_title(f"Training {algorithm} - Episode {ep}")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            plt.tight_layout()
            
            # YIELD the figure to update Gradio instantly
            yield fig 
            plt.close(fig)

# Build the UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Live Actor-Critic Training Dashboard")
    
    with gr.Row():
        algo_drop = gr.Dropdown(["REINFORCE", "A2C"], value="A2C", label="Algorithm")
        lr_slider = gr.Slider(0.001, 0.05, value=0.01, label="Learning Rate")
        ep_slider = gr.Slider(50, 500, value=200, step=10, label="Episodes")
    
    train_btn = gr.Button("▶️ Start Training")
    plot_output = gr.Plot(label="Live Training Curve")
    
    # Map the button to the generator function
    train_btn.click(train_agent_live, inputs=[algo_drop, lr_slider, ep_slider], outputs=plot_output)

demo.queue().launch()
```

---

### Deployment to Hugging Face Spaces

1. **Create Space**: Ensure the SDK is set to **Gradio**.
2. **Requirements**: Your `requirements.txt` MUST include:
   ```text
   gradio
   tensorflow
   gymnasium[classic_control]
   matplotlib
   numpy
   ```
3. **Commit**: Push your `app.py` and `requirements.txt`.
4. **Hardware Note**: Spaces run on CPUs by default. Deep RL training on CPUs is fine for simple environments like CartPole! Ensure your episode count slider is capped at a reasonable number (e.g., max 500) so Hugging Face doesn't time out.

> 💡 **Why This Matters**: Live-streaming data from a long-running backend process (training) to a frontend dashboard (Gradio) involves dealing with WebSockets and asynchronous generators. Adding this to your portfolio demonstrates you can build full-stack ML applications, not just Jupyter Notebooks.
