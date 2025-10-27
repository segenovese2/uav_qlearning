import numpy as np
import random
import json

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.95, exploration_rate=1.0, 
                 exploration_decay=0.998, min_exploration_rate=0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(low=-0.1, high=0.1, size=(state_size, action_size))
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_episodes = []
        self.communication_qualities = []
        
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, terminated):
        """Update Q-table using Q-learning update rule"""
        current_q = self.q_table[state, action]
        
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filepath):
        """Save Q-table to file"""
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")
    
    def load_q_table(self, filepath):
        """Load Q-table from file"""
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from {filepath}")
    
    def save_training_stats(self, filepath):
        """Save training statistics to JSON file"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successful_episodes': self.successful_episodes,
            'communication_qualities': self.communication_qualities,
            'final_epsilon': self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f)
        print(f"Training stats saved to {filepath}")
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successful_episodes': self.successful_episodes,
            'communication_qualities': self.communication_qualities,
            'epsilon': self.epsilon
        }
