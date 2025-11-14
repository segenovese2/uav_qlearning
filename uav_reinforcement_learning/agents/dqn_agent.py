import numpy as np
import random
import json
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Deep Q-Network for UAV trajectory optimization"""
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    Deep Q-Network Agent compatible with Gymnasium environments.
    Drop-in replacement for QLearningAgent with the same interface.
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        batch_size=64,
        memory_size=10000,
        target_update_freq=10,
        hidden_sizes=[128, 128, 64]
    ):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial epsilon for epsilon-greedy
            exploration_decay: Decay rate for epsilon
            min_exploration_rate: Minimum epsilon value
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            target_update_freq: Frequency (episodes) to update target network
            hidden_sizes: List of hidden layer sizes
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.policy_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics (compatible with Q-learning agent interface)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_episodes = []
        self.communication_qualities = []
        self.reached_midpoint_count = []
        self.returned_to_base_count = []
        
        # Internal counters
        self.episodes_since_update = 0
        self.total_steps = 0
        
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state (numpy array)
            training: Whether in training mode (enables exploration)
            
        Returns:
            action: Selected action (int)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, terminated):
        """
        Store transition and train the network
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            terminated: Whether episode terminated
        """
        # Store transition in replay memory
        self.memory.append((state, action, reward, next_state, terminated))
        self.total_steps += 1
        
        # Train if enough samples in memory
        if len(self.memory) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """Perform one training step using a batch from replay memory"""
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        terminateds = torch.FloatTensor(terminateds).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - terminateds) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def decay(self):
        """
        Decay exploration rate and update target network periodically.
        Called at the end of each episode.
        """
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.episodes_since_update += 1
        if self.episodes_since_update >= self.target_update_freq:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.episodes_since_update = 0
    
    def save(self, filepath):
        """
        Save model weights to file
        
        Args:
            filepath: Path to save the model (will append .pt extension)
        """
        if not filepath.endswith('.pt'):
            filepath = filepath + '.pt'
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }, filepath)
        print(f"DQN model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model weights from file
        
        Args:
            filepath: Path to load the model from
        """
        if not filepath.endswith('.pt'):
            filepath = filepath + '.pt'
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        print(f"DQN model loaded from {filepath}")
    
    def save_training_stats(self, filepath):
        """
        Save training statistics to JSON file
        
        Args:
            filepath: Path to save the statistics
        """
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successful_episodes': self.successful_episodes,
            'communication_qualities': self.communication_qualities,
            'reached_midpoint_count': self.reached_midpoint_count,
            'returned_to_base_count': self.returned_to_base_count,
            'final_epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'memory_size': len(self.memory)
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Training stats saved to {filepath}")
    
    def get_q_values(self, state):
        """
        Get Q-values for all actions in a given state (for analysis)
        
        Args:
            state: State to evaluate
            
        Returns:
            Q-values for all actions (numpy array)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]
