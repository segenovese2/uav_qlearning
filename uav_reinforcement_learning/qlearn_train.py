import gymnasium as gym
import numpy as np
import pygame
from environments.uav_env import UAVEnv
from agents.q_learning_agent import QLearningAgent
from utils.plotter import plot_training_stats

def calculate_midpoint(users):
    """Calculate midpoint between two users"""
    return np.mean(users, axis=0)

def train_q_learning(show_training=True):
    render_mode = "human" if show_training else None
    env = UAVEnv(grid_size=15, render_mode=render_mode)
    
    state_size = 5
    action_size = env.action_space.n
    
    # Calculate target midpoint
    midpoint = calculate_midpoint(env.users)
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Grid: {env.grid_size}x{env.grid_size}")
    print(f"Users: {env.users}")
    print(f"Target Midpoint: ({midpoint[0]:.1f}, {midpoint[1]:.1f})")
    print(f"Max steps: {env.max_steps}")
    
    # Create Q-learning agent with adjusted parameters
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.2,  # Higher for faster updates
        discount_factor=0.98,  # Very high for long-term planning
        exploration_rate=1.0,
        exploration_decay=0.9995,  # Very slow decay
        min_exploration_rate=0.1  # Keep exploring
    )
    
    num_episodes = 15000
    print_interval = 100
    success_count = 0
    midpoint_reached_count = 0
    returned_to_base_count = 0
    
    print("\nStarting Q-learning training for UAV trajectory optimization...")
    print("Goal: Fly to midpoint between users, then return to base (0,0)")
    print("Close the Pygame window to stop training early\n")
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            episode_comm_quality = 0
            
            reached_midpoint = False
            closest_to_midpoint = float('inf')
            max_distance_from_base = 0
            
            while not (terminated or truncated):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Enhanced reward shaping with STRONG distance-based guidance
                current_pos = env.current_pos
                step_ratio = steps / env.max_steps
                
                # Distance to midpoint
                dist_to_midpoint = np.linalg.norm(current_pos - midpoint)
                closest_to_midpoint = min(closest_to_midpoint, dist_to_midpoint)
                
                # Distance to base
                dist_to_base = np.linalg.norm(current_pos - env.goal_pos)
                max_distance_from_base = max(max_distance_from_base, dist_to_base)
                
                # Start with communication quality (scaled up significantly)
                comm_quality = info['communication_quality']
                shaped_reward = comm_quality * 2.0  # Much stronger base reward
                
                # STRONG penalties for being in bad communication zones
                if list(current_pos) in env.nlos_both:
                    shaped_reward -= 5.0  # Heavy penalty for no line of sight to both users
                elif list(current_pos) in env.nlos_single:
                    shaped_reward -= 2.0  # Moderate penalty for partial NLOS
                
                # Bonus for good communication
                if comm_quality > 0.7:
                    shaped_reward += 3.0  # Extra reward for excellent signal
                
                # CRITICAL: Phase-based rewards with strong distance incentives
                if step_ratio < 0.5:
                    # EARLY PHASE: Must leave base area and head toward midpoint
                    
                    # Penalty for staying near base
                    if dist_to_base < 3.0:
                        shaped_reward -= 2.0
                    
                    # BIG reward for moving away from base
                    shaped_reward += dist_to_base * 1.5
                    
                    # HUGE reward for getting close to midpoint
                    if dist_to_midpoint < 5.0:
                        shaped_reward += (5.0 - dist_to_midpoint) * 5.0
                    
                    # MASSIVE bonus for reaching midpoint
                    if dist_to_midpoint < 2.5:
                        shaped_reward += 50.0
                        reached_midpoint = True
                        
                else:
                    # LATE PHASE: Must return to base
                    
                    # Penalty for staying far from base
                    if dist_to_base > 5.0:
                        shaped_reward -= 3.0
                    
                    # BIG reward for getting closer to base
                    shaped_reward += (15.0 - dist_to_base) * 2.0
                    
                    # Extra bonus if we previously reached midpoint
                    if reached_midpoint:
                        shaped_reward += 5.0
                        # Even bigger reward for being close to base after visiting midpoint
                        if dist_to_base < 3.0:
                            shaped_reward += 15.0
                
                # Strong obstacle penalty
                if info['hit_obstacle']:
                    shaped_reward -= 10.0
                
                # Update Q-table with shaped reward
                agent.update(state, action, shaped_reward, next_state, terminated)
                
                state = next_state
                total_reward += shaped_reward
                steps += 1
                episode_comm_quality += info['communication_quality']
                
                if show_training:
                    env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("Training interrupted by user")
                            env.close()
                            return agent
            
            # Final evaluation at end of episode
            final_pos = env.current_pos
            dist_to_base_final = np.linalg.norm(final_pos - env.goal_pos)
            
            # Success criteria
            reached_near_midpoint = closest_to_midpoint < 3.0
            returned_to_base = dist_to_base_final < 2.0
            explored_away = max_distance_from_base > 8.0  # Did we actually leave base area?
            
            # End-of-episode bonuses
            if explored_away:
                total_reward += 30.0
                
            if reached_near_midpoint:
                midpoint_reached_count += 1
                total_reward += 50.0
                
            if returned_to_base:
                returned_to_base_count += 1
                total_reward += 40.0
            
            # HUGE bonus for complete success
            success = reached_near_midpoint and returned_to_base and explored_away
            if success:
                success_count += 1
                total_reward += 100.0
            
            avg_comm_quality = episode_comm_quality / steps if steps > 0 else 0
            
            # Store statistics
            agent.episode_rewards.append(total_reward)
            agent.episode_lengths.append(steps)
            agent.successful_episodes.append(success_count)
            agent.communication_qualities.append(avg_comm_quality)
            agent.reached_midpoint_count.append(midpoint_reached_count)
            agent.returned_to_base_count.append(returned_to_base_count)
            
            agent.decay()
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(agent.episode_rewards[-print_interval:])
                avg_length = np.mean(agent.episode_lengths[-print_interval:])
                avg_comm = np.mean(agent.communication_qualities[-print_interval:])
                success_rate = (success_count / (episode + 1)) * 100
                midpoint_rate = (midpoint_reached_count / (episode + 1)) * 100
                return_rate = (returned_to_base_count / (episode + 1)) * 100
                
                # Calculate recent performance
                recent_success = success_count - agent.successful_episodes[max(0, episode - print_interval)]
                recent_midpoint = midpoint_reached_count - agent.reached_midpoint_count[max(0, episode - print_interval)]
                recent_return = returned_to_base_count - agent.returned_to_base_count[max(0, episode - print_interval)]
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.1f} | Avg Steps: {avg_length:.1f} | Epsilon: {agent.epsilon:.3f}")
                print(f"  Overall - Success: {success_rate:.1f}% | Midpoint: {midpoint_rate:.1f}% | Return: {return_rate:.1f}%")
                print(f"  Last 100 - Success: {recent_success} | Midpoint: {recent_midpoint} | Return: {recent_return}")
                print(f"  Last Episode - Closest to Mid: {closest_to_midpoint:.1f} | Max Dist from Base: {max_distance_from_base:.1f}")
                print()
            
            # Detailed debug for early episodes
            if episode < 5 or (episode + 1) % 500 == 0:
                print(f"Episode {episode + 1} Detail:")
                print(f"  Final Pos: {final_pos} | Dist to Base: {dist_to_base_final:.1f}")
                print(f"  Closest to Midpoint: {closest_to_midpoint:.1f} | Max Dist from Base: {max_distance_from_base:.1f}")
                print(f"  Reached Midpoint: {reached_near_midpoint} | Explored: {explored_away} | Returned: {returned_to_base}")
                print(f"  Total Reward: {total_reward:.1f}\n")
        
        # Save results
        agent.save('trained_q_table.npy')
        agent.save_training_stats('training_stats.json')
        
        plot_training_stats(
            agent.episode_rewards, 
            agent.episode_lengths, 
            agent.successful_episodes,
            agent.communication_qualities
        )
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Final Success Rate: {(success_count / num_episodes) * 100:.1f}%")
        print(f"Final Midpoint Rate: {(midpoint_reached_count / num_episodes) * 100:.1f}%")
        print(f"Final Return Rate: {(returned_to_base_count / num_episodes) * 100:.1f}%")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save('trained_q_table_interrupted.npy')
        agent.save_training_stats('training_stats_interrupted.json')
    finally:
        env.close()
    
    return agent

if __name__ == "__main__":
    agent = train_q_learning(show_training=True)
