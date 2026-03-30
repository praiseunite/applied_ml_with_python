import os
# Suppress TensorFlow logging to keep console clean for students
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import tensorflow as as_tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers
import gymnasium as gym
import matplotlib.pyplot as plt

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

class REINFORCEAgent:
    """
    Monte Carlo Policy Gradient Agent (REINFORCE)
    Learns a parameterised policy (Actor) without a value function.
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Build the Policy Network
        self.model = self._build_model()
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        """Creates a simple neural network to approximate the policy."""
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_dim=self.state_dim),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_dim, activation='softmax') # Outputs action probabilities!
        ])
        return model

    def choose_action(self, state):
        """Sample an action probabilistically from the policy network."""
        # state is shape (n,) -> reshape to (1, n) for TF
        state_tensor = as_tf.convert_to_tensor([state], dtype=as_tf.float32)
        action_probs = self.model(state_tensor).numpy()[0]
        
        # Sample action based on the probabilities given by the network
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def calculate_returns(self, rewards):
        """
        Calculate discounted cumulative returns (G_t).
        If reward was [1, 1, 1], returns are [1 + γ + γ², 1 + γ, 1].
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
            
        # Normalize returns (Standardization) to reduce variance for stability
        # A common trick in Deep RL!
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns

    def train_step(self, states, actions, returns):
        """Performs one gradient ascent step using a full episode's data."""
        states = as_tf.convert_to_tensor(states, dtype=as_tf.float32)
        actions = as_tf.convert_to_tensor(actions, dtype=as_tf.int32)
        returns = as_tf.convert_to_tensor(returns, dtype=as_tf.float32)
        
        with as_tf.GradientTape() as tape:
            # Predict action probabilities
            probs = self.model(states)
            
            # Create a mask to select only the probabilities of the actions actually taken
            indices = as_tf.stack([as_tf.range(len(actions)), actions], axis=-1)
            selected_probs = as_tf.gather_nd(probs, indices)
            
            # Loss = -1 * sum(log_prob * return)
            # We multiply by -1 because TensorFlow minimizes loss, but we want to MAXIMIZE returns (Gradient Ascent)
            loss = as_tf.reduce_sum(-as_tf.math.log(selected_probs + 1e-8) * returns)
            
        # Compute and apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()

def main():
    print_header("REINFORCE (Policy Gradient) on CartPole-v1")
    
    # 1. Setup Environment
    try:
        env = gym.make('CartPole-v1')
        print("✅ Gymnasium Environment 'CartPole-v1' Loaded.")
    except Exception as e:
        print(f"❌ Failed to load Gymnasium: {e}")
        sys.exit(1)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 2. Initialize Agent
    agent = REINFORCEAgent(state_dim, action_dim, learning_rate=0.01)
    print("✅ Policy Network (Actor) Built Successfully.")
    
    # 3. Training Loop
    # Wait until episode ends to calculate returns (Monte Carlo)
    EPISODES = 300
    reward_history = []
    
    print("\n🚀 Starting Training... (Target: 500 reward limit)")
    for ep in range(EPISODES):
        state, _ = env.reset()
        episode_states, episode_actions, episode_rewards = [], [], []
        done = False
        
        # Play the game using the current policy
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
            
        total_reward = sum(episode_rewards)
        reward_history.append(total_reward)
        
        # Time to learn! (Only after the episode finishes)
        returns = agent.calculate_returns(episode_rewards)
        agent.train_step(episode_states, episode_actions, returns)
        
        # Logging
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(reward_history[-20:])
            print(f"Episode {ep + 1:3d}/{EPISODES} | Last 20 Avg Reward: {avg_reward:5.1f}")
            if avg_reward >= 490:
                print("\n🎯 Converged! Agent mastered CartPole.")
                break
                
    env.close()
    print("\n🏁 Training Complete.")
    
    # 4. Visualize Learning
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Episode Reward', color='#3498db', alpha=0.5)
    
    # Calculate moving average (window=20)
    window = 20
    if len(reward_history) > window:
        moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(reward_history)), moving_avg, 
                 label=f'{window}-Episode Moving Avg', color='#2ecc71', linewidth=2)
                 
    plt.title('REINFORCE Learning Curve on CartPole-v1', fontweight='bold', fontsize=14)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
