# Session 04 — Exercises

## 🟢 Beginner Exercises

### Exercise 4.1: Calculating Returns (The Math of REINFORCE)

**Objective**: Understand how future rewards are attributed to past actions.

**Instructions**:
An agent plays an episode that lasts for 5 steps. It receives the following sequence of immediate rewards: `[0, 0, 0, 10, -5]`

**Part A**: Calculate the Discounted Return $G_t$ for each of the 5 steps using a discount factor of $\gamma = 0.9$.
*(Hint: calculate backwards from the last step)*

**Part B**: What happens to $G_t$ at step 0 if $\gamma = 0$? What happens if $\gamma = 1$?

---

### Exercise 4.2: Building the Actor Network 

**Objective**: Use TensorFlow/Keras to build a policy network capable of handling continuous state inputs.

**Instructions**:
Write a function `build_actor(state_dim, action_dim)` using `tensorflow.keras`.
The network should:
1. Take an input of size `state_dim`.
2. Have two hidden layers with 64 neurons each and 'relu' activation.
3. Output a layer of size `action_dim` with an activation function that guarantees the outputs sum to exactly `1.0`.

**Starter Code**:
```python
import tensorflow.keras as keras
from tensorflow.keras import layers

def build_actor(state_dim, action_dim):
    # YOUR CODE HERE
    pass
```

---

## 🟡 Intermediate Exercises

### Exercise 4.3: Variance Reduction via Baselines

**Objective**: Prove mathematically how a baseline reduces the variance of the return gradients without changing their expected value.

**Instructions**:
You run REINFORCE for 3 episodes. The total return $G_0$ for each episode is:
- Episode 1: 500
- Episode 2: 520
- Episode 3: 490

**Part A**: Calculate the Variance of these three returns. (You can use `np.var()`).
**Part B**: Assume a Critic Network perfectly predicted the average return as its baseline, so $V(s_0) = 503.33$. Calculate the **Advantage** $(G_0 - V(s_0))$ for each episode.
**Part C**: Calculate the Variance of those three Advantages. Does the variance change? Why does this make training a neural network more stable?

---

### Exercise 4.4: The Policy Gradient Loss Function

**Objective**: Understand why the REINFORCE loss function is written the way it is in TensorFlow.

**Instructions**:
In standard supervised learning (like cat vs dog classification), we use `CategoricalCrossentropy` loss. 
In REINFORCE, we calculated loss manually:
```python
loss = -as_tf.math.log(prob_of_action) * Return
```

Why do we multiply by the `Return`? What would happen if the agent took an action that resulted in a highly *negative* return (e.g., `-100`), and what does the `-` sign at the front of the equation do? Draft a 100-word explanation.

---

## 🔴 Advanced Exercises

### Exercise 4.5: Continuous Action Spaces 

**Objective**: Modify a Policy Network to output continuous actions (like steering a car) instead of discrete buttons.

**Instructions**:
If an environment accepts a continuous float between `-1.0` and `1.0` (like steering), we cannot use a `softmax` output layer. Instead, the network must output the parameters of a **Normal (Gaussian) Distribution**: a mean ($\mu$) and a standard deviation ($\sigma$).

1. Build a Keras model that outputs **two** values: `mu` (using `tanh` activation to bound it between -1 and +1) and `sigma` (using `softplus` activation to ensure it is always positive).
2. Write a function that takes `mu` and `sigma`, and samples a single float value from that Gaussian distribution using `np.random.normal()`.

---

### Exercise 4.6: Implement Gradient Clipping (PPO Concept)

**Objective**: Implement safety bounds on neural network updates.

**Instructions**:
In A2C and PPO, massive advantages can result in massive gradients, which destroy the network's weights. 
Using the `02_actor_critic_cartpole.py` script as a base, locate the `tape_actor.gradient()` step. 

Modify the script so that before the gradients are applied using `apply_gradients()`, all gradient values are **clipped** so they cannot exceed `[-1.0, 1.0]`. 
*(Hint: look up `tf.clip_by_value` or `tf.clip_by_norm` in the TensorFlow documentation)*. Run the script and observe if it stabilizes the learning curve.

---

## 📝 Submission Guidelines
- Submit code as Python scripts (`.py`) or Jupyter Notebooks (`.ipynb`).
- Written answers should be concise but mathematically accurate.
- Name your files `exercise_4_X.py`.
