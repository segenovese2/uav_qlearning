import gymnasium as gym
import numpy as np
import pygame
from environments.uav_env import UAVEnv
from agents.q_learning_agent import QLearningAgent
from utils.plotter import plot_training_stats

def train_q_learning(show_training=True):
    # Create environment with rendering
    render_mode = "human" if show_training else None
    env = UAVEnv(grid_size=15, render_mode=render_mode)
    
    # Calculate state size for Q-table
    state_size = env.grid_size * env.grid_size
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Grid: {env.grid_size}x{env.grid_size}")
    print(f"Obstacles: {len(env.obstacles)}")
    print(f"Users: {len(env.users)} at positions {env.users}")
    print(f"Max steps: {env.max_steps}")
    
    # Create Q-learning agent
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.998,
        min_exploration_rate=0.01
    )
    
    # Training parameters
    num_episodes = 10000
    print_interval = 100
    success_count = 0
    
    print("Starting Q-learning training for UAV trajectory optimization...")
    print("Close the Pygame window to stop training early")
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            state_idx = env.get_state_index(state)
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            episode_comm_quality = 0
            
            while not (terminated or truncated):
                # Choose action
                action = agent.choose_action(state_idx)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state_idx = env.get_state_index(next_state)
                
                # Update Q-table
                agent.update(state_idx, action, reward, next_state_idx, terminated)
                
                state_idx = next_state_idx
                total_reward += reward
                steps += 1
                episode_comm_quality += info['communication_quality']
                
                # Render the environment
                if show_training:
                    env.render()
                    
                    # Check for Pygame quit event
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("Training interrupted by user")
                            env.close()
                            return agent
            
            # Calculate average communication quality
            avg_comm_quality = episode_comm_quality / steps if steps > 0 else 0
            
            # Check if episode was successful (reached goal at step 50)
            success = (terminated and np.array_equal(env.current_pos, env.goal_pos) and steps == env.max_steps)
            if success:
                success_count += 1
            
            # Store episode statistics
            agent.episode_rewards.append(total_reward)
            agent.episode_lengths.append(steps)
            agent.successful_episodes.append(success_count)
            agent.communication_qualities.append(avg_comm_quality)
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(agent.episode_rewards[-print_interval:])
                avg_length = np.mean(agent.episode_lengths[-print_interval:])
                avg_comm = np.mean(agent.communication_qualities[-print_interval:])
                success_rate = (success_count / (episode + 1)) * 100
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}, "
                      f"Avg Comm: {avg_comm:.2f}, "
                      f"Success Rate: {success_rate:.1f}%, "
                      f"Epsilon: {agent.epsilon:.3f}")
                
                # Reset success count for this interval
                success_count = 0
        
        # Save trained Q-table and statistics
        agent.save_q_table('trained_q_table.npy')
        agent.save_training_stats('training_stats.json')
        
        # Plot training results
        plot_training_stats(
            agent.episode_rewards, 
            agent.episode_lengths, 
            agent.successful_episodes,
            agent.communication_qualities
        )
        
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save progress even if interrupted
        agent.save_q_table('trained_q_table_interrupted.npy')
        agent.save_training_stats('training_stats_interrupted.json')
    finally:
        env.close()
    
    return agent

if __name__ == "__main__":
    # Set show_training=True to see real-time visualization
    agent = train_q_learning(show_training=True)
