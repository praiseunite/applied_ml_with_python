import os
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

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent.
    - Actor: Decides which action to take.
    - Critic: Evaluates the state to provide a baseline (reduces variance).
    """
    def __init__(self, state_dim, action_dim, lr_actor=0.005, lr_critic=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Build independent Actor and Critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = optimizers.Adam(learning_rate=lr_critic)

    def _build_actor(self):
        """Policy Network: State -> Action Probabilities"""
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_dim=self.state_dim),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def _build_critic(self):
        """Value Network: State -> Expected Future Reward (V value)"""
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_dim=self.state_dim),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear') # Single continuous value output!
        ])
        return model

    def choose_action(self, state):
        """Sample an action probabilistically from the Actor."""
        state_tensor = as_tf.convert_to_tensor([state], dtype=as_tf.float32)
        action_probs = self.actor(state_tensor).numpy()[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def train_step(self, state, action, reward, next_state, done):
        """
        Unlike REINFORCE, Actor-Critic learns at EVERY step (Temporal Difference learning),
        not waiting until the end of the episode!
        """
        state = as_tf.convert_to_tensor([state], dtype=as_tf.float32)
        next_state = as_tf.convert_to_tensor([next_state], dtype=as_tf.float32)
        reward = as_tf.convert_to_tensor(reward, dtype=as_tf.float32)
        
        with as_tf.GradientTape() as tape_critic:
            # Critic predicts Value of current state
            V_s = self.critic(state)[0, 0]
            # Critic predicts Value of next state
            V_next = self.critic(next_state)[0, 0]
            
            # Target = r + gamma * V(s')
            # If done, there is no future value.
            target = reward + (1.0 - float(done)) * self.gamma * V_next
            
            # The Advantage (A) = actual outcome - expected outcome
            # The critic learns to minimize this error (MSE).
            advantage = target - V_s
            critic_loss = as_tf.square(advantage)

        # Update Critic
        critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # We detach 'advantage' so it acts as a constant scalar for the Actor 
        # (the Actor doesn't train the Critic's weights).
        adv_val = as_tf.stop_gradient(advantage)

        with as_tf.GradientTape() as tape_actor:
            probs = self.actor(state)[0] # e.g. [0.2, 0.8]
            prob_of_action = probs[action]
            
            # Actor Loss = -log(prob) * Advantage
            # If advantage is positive (good!), loss goes deeply negative (gradient ascent pushes prob UP).
            actor_loss = -as_tf.math.log(prob_of_action + 1e-8) * adv_val
            
        # Update Actor
        actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss.numpy(), critic_loss.numpy()

def main():
    print_header("Advantage Actor-Critic (A2C) on CartPole-v1")
    
    try:
        env = gym.make('CartPole-v1')
    except Exception as e:
        print(f"❌ Failed to load Gymnasium: {e}")
        sys.exit(1)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2CAgent(state_dim, action_dim, lr_actor=0.002, lr_critic=0.005)
    print("✅ Actor (Policy) Network Built.")
    print("✅ Critic (Value) Network Built.")
    
    EPISODES = 300
    reward_history = []
    
    print("\n🚀 Starting Training... (You will notice it learns faster but stochastically!)")
    for ep in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # A2C learns immediately after taking the action!
            agent.train_step(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
        reward_history.append(total_reward)
        
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(reward_history[-20:])
            print(f"Episode {ep + 1:3d}/{EPISODES} | Last 20 Avg Reward: {avg_reward:5.1f}")
            if avg_reward >= 490:
                print("\n🎯 Converged! Actor-Critic mastered CartPole.")
                break
                
    env.close()
    
    # Visualizing the Learning Process
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Episode Reward', color='#e74c3c', alpha=0.5)
    window = 20
    if len(reward_history) > window:
        moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(reward_history)), moving_avg, 
                 label=f'{window}-Episode Moving Avg', color='#c0392b', linewidth=2)
                 
    plt.title('A2C Learning Curve on CartPole-v1', fontweight='bold', fontsize=14)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
