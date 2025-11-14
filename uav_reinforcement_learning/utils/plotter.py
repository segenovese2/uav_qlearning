import matplotlib.pyplot as plt
import numpy as np

def plot_training_stats(episode_rewards, communication_qualities, window=100):
    """Plot training statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    rewards = np.array(episode_rewards)
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    ax1.plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
    ax1.plot(moving_avg, label=f'Moving Average ({window} episodes)', color='red', linewidth=2)
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
        
    # Plot communication quality
    comm_qualities = np.array(communication_qualities)
    comm_avg = np.convolve(comm_qualities, np.ones(window)/window, mode='valid')
    
    ax4.plot(comm_qualities, alpha=0.3, label='Comm Quality', color='orange')
    ax4.plot(comm_avg, label=f'Moving Average ({window} episodes)', color='red', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Communication Quality')
    ax4.set_title('Average Communication Quality')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('uav_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_q_table(q_table, grid_size):
    """Visualize the Q-table as a heatmap for each action"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    # Find global min and max for consistent coloring
    vmin = np.min(q_table)
    vmax = np.max(q_table)
    
    for action in range(4):
        ax = axes[action // 2, action % 2]
        q_values = q_table[:, action].reshape(grid_size, grid_size)
        
        im = ax.imshow(q_values, cmap='RdYlGn', interpolation='nearest', 
                      vmin=vmin, vmax=vmax)
        ax.set_title(f'Q-values for Action: {action_names[action]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add value annotations for important cells
        for i in range(min(grid_size, 15)):  # Limit to first 15x15 for readability
            for j in range(min(grid_size, 15)):
                if abs(q_values[i, j]) > 0.1:  # Only show significant values
                    text = ax.text(j, i, f'{q_values[i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('uav_q_table_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
