# Session 04 — Solutions

## 🟢 Solution 4.1: Calculating Returns

**Part A (Discounted Returns):**
Rewards: `[0, 0, 0, 10, -5]` with $\gamma = 0.9$

We calculate backwards from $t=4$:
- $G_4$: `-5`
- $G_3$: `10 + (0.9 * -5) = 10 - 4.5 = 5.5`
- $G_2$: `0 + (0.9 * 5.5) = 4.95`
- $G_1$: `0 + (0.9 * 4.95) = 4.455`
- $G_0$: `0 + (0.9 * 4.455) = 4.0095`

Returns = `[4.0095, 4.455, 4.95, 5.5, -5]`

**Part B:**
- If $\gamma = 0$: $G_t$ only considers the immediate reward. So $G_0 = 0$. The agent is severely short-sighted.
- If $\gamma = 1$: $G_t$ treats all future rewards equally. $G_0 = 0 + 0 + 0 + 10 - 5 = 5$.

---

## 🟢 Solution 4.2: Building the Actor Network

```python
import tensorflow.keras as keras
from tensorflow.keras import layers

def build_actor(state_dim, action_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=state_dim),
        layers.Dense(64, activation='relu'),
        # 'softmax' guarantees the values sum to 1.0 (probabilities)
        layers.Dense(action_dim, activation='softmax')
    ])
    return model
```

---

## 🟡 Solution 4.3: Variance Reduction via Baselines

**Part A (Variance of Raw Returns):**
Returns: `[500, 520, 490]`
Mean = 503.33
Variance = `((500 - 503.33)^2 + (520 - 503.33)^2 + (490 - 503.33)^2) / 3` ≈ `155.55`

**Part B (Advantages):**
$V(s) = 503.33$
- $A_1$: `500 - 503.33 = -3.33` (Worse than expected)
- $A_2$: `520 - 503.33 = 16.67` (Better than expected)
- $A_3$: `490 - 503.33 = -13.33` (Worse than expected)

**Part C (Variance and Stability):**
The variance of `[-3.33, 16.67, -13.33]` is still `155.55`. 
*Wait, the variance didn't change!* Why use a baseline? 

**The Insight**: The variance of the *returns* themselves doesn't change by subtracting a constant. However, the variance of the **Gradient Updates** changes dramatically. Because the updates are proportional to the magnitude of the advantage, multiplying the gradients by small numbers `[-3.33, 16.67]` instead of massive numbers `[500, 520]` prevents the neural network weights from taking massive, erratic jumps. It centers the learning signal around 0.

---

## 🟡 Solution 4.4: The Policy Gradient Loss Function

**Explanation:**
In TensorFlow, optimizers like Adam always perform Gradient **Descent** (they minimize the loss). We want to maximize rewards, so we multiply by `-1` (Gradient **Ascent**).

`loss = -log(prob) * Return`
- If the agent takes a good action (`Return = +100`), the loss becomes highly negative. TensorFlow updates the weights to make this state output the exact same action more often (pushing probability toward 1.0).
- If `Return = -100` (terrible action), the loss becomes highly positive. TensorFlow updates the weights to *avoid* outputting this action again (pushing probability toward 0.0).

---

## 🔴 Solution 4.5: Continuous Action Spaces

```python
import numpy as np
import tensorflow as as_tf
from tensorflow.keras import layers, Model, Input

def build_continuous_actor(state_dim):
    inputs = Input(shape=(state_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    
    # Mean of the distribution (e.g. steering angle -1 to 1)
    mu = layers.Dense(1, activation='tanh', name='mean')(x)
    
    # Standard Deviation (must be positive, so we use softplus)
    sigma = layers.Dense(1, activation='softplus', name='std_dev')(x)
    
    return Model(inputs=inputs, outputs=[mu, sigma])

def sample_continuous_action(mu, sigma):
    # Sample from the Gaussian distribution defined by the network
    action = np.random.normal(loc=mu, scale=sigma)
    return action

# Test
actor = build_continuous_actor(4)
mu, sigma = actor(np.array([[0,0,0,0]]))
action = sample_continuous_action(mu.numpy()[0][0], sigma.numpy()[0][0])
print(f"Sampled continuous action: {action:.3f}")
```

---

## 🔴 Solution 4.6: Implement Gradient Clipping

**Modified A2C Training Step (Excerpt):**

```python
with as_tf.GradientTape() as tape_actor:
    probs = self.actor(state)[0] 
    prob_of_action = probs[action]
    actor_loss = -as_tf.math.log(prob_of_action + 1e-8) * adv_val

# Get gradients
actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)

# CLIP GRADIENTS! (This stops the network from exploding)
clipped_actor_grads = [as_tf.clip_by_norm(g, 1.0) for g in actor_grads]

# Apply clipped gradients
self.actor_optimizer.apply_gradients(zip(clipped_actor_grads, self.actor.trainable_variables))
```

**Observation**: You will notice that the reward curve becomes significantly smoother and is much less likely to "forget" how to play CartPole after it has learned it.
