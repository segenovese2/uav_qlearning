import numpy as np
import random
import json

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.95, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        
        # Enhanced state discretization
        self.grid_size = 15
        self.phase_bins = 3  # early, mid, late phase of flight
        
        # State space: position (15x15) + phase (3) = 675 states
        total_states = self.grid_size * self.grid_size * self.phase_bins
        
        # Initialize Q-table with optimistic values to encourage exploration
        self.q_table = np.random.uniform(low=0, high=0.5, size=(total_states, action_size))
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_episodes = []
        self.communication_qualities = []
        self.reached_midpoint_count = []
        self.returned_to_base_count = []
        
    def _get_state_index(self, state):
        """Convert continuous state to discrete index including phase information"""
        # Extract from state vector [uav_x, uav_y, user1_dist, user2_dist, step_ratio]
        uav_x = state[0]
        uav_y = state[1]
        step_ratio = state[4]
        
        # Discretize position
        grid_x = int(uav_x * (self.grid_size - 1))
        grid_y = int(uav_y * (self.grid_size - 1))
        
        # Discretize phase (0: early 0-0.4, 1: mid 0.4-0.7, 2: late 0.7-1.0)
        if step_ratio < 0.4:
            phase = 0  # Should be exploring/moving to midpoint
        elif step_ratio < 0.7:
            phase = 1  # Should be near midpoint
        else:
            phase = 2  # Should be returning to base
        
        # Combine into single index
        pos_idx = grid_x * self.grid_size + grid_y
        state_idx = pos_idx * self.phase_bins + phase
        
        return min(state_idx, self.grid_size * self.grid_size * self.phase_bins - 1)
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        state_idx = self._get_state_index(state)
        
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state, terminated):
        """Update Q-table using Q-learning update rule"""
        state_idx = self._get_state_index(state)
        next_state_idx = self._get_state_index(next_state)
        
        current_q = self.q_table[state_idx, action]
        
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx, action] = current_q + self.lr * (target - current_q)
    
    def decay(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save Q-table to file"""
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath):
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
            'reached_midpoint_count': self.reached_midpoint_count,
            'returned_to_base_count': self.returned_to_base_count,
            'final_epsilon': self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f)
        print(f"Training stats saved to {filepath}")
