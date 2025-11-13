# [file name]: uav_qlearning/agents/qnet_agent.py
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Simple neural network for Q-learning"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        
        # Simpler 2-layer network for Q-Network
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on output


class QNetAgent:
    """
    Q-Network Agent: Neural network Q-learning without experience replay or target network.
    This is simpler than DQN but more powerful than tabular Q-learning.
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 discount_factor=0.98, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Q-Network (single network, no target network)
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_episodes = []
        self.communication_qualities = []
        self.reached_midpoint_count = []
        self.returned_to_base_count = []
        self.losses = []
        
        # Training step counter
        self.training_steps = 0
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action from network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-network using standard Q-learning update.
        No experience replay - learns directly from each transition.
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # Current Q value for the taken action
        current_q = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze()
        
        # Next Q value (max over actions)
        with torch.no_grad():
            next_q = self.q_network(next_state_tensor).max(1)[0]
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save the Q-network"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the Q-network"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        print(f"Model loaded from {filepath}")
    
    def save_training_stats(self, filepath):
        """Save training statistics to JSON file"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successful_episodes': self.successful_episodes,
            'communication_qualities': self.communication_qualities,
            'reached_midpoint_count': self.reached_midpoint_count,
            'returned_to_base_count': self.returned_to_base_count,
            'losses': self.losses,
            'final_epsilon': self.epsilon,
            'training_steps': self.training_steps
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f)
        print(f"Training stats saved to {filepath}")
