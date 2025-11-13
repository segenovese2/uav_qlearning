import gymnasium as gym
import numpy as np
from environments.uav_env import UAVEnv
from agents.q_learning_agent import QLearningAgent
from utils.plotter import plot_training_stats
import time

def calculate_midpoint(users):
    """Calculate midpoint between two users"""
    return np.mean(users, axis=0)

def train_q_learning_fast():
    """Fast training with no rendering"""
    # NO RENDER MODE - this is the key to speed
    env = UAVEnv(grid_size=15, render_mode=None)
    
    state_size = 5
    action_size = env.action_space.n
    midpoint = calculate_midpoint(env.users)
    
    print(f"=== FAST TRAINING MODE (No Rendering) ===")
    print(f"Grid: {env.grid_size}x{env.grid_size}")
    print(f"Users: {env.users}")
    print(f"Target Midpoint: ({midpoint[0]:.1f}, {midpoint[1]:.1f})")
    print(f"Max steps: {env.max_steps}\n")
    
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.2,
        discount_factor=0.98,
        exploration_rate=1.0,
        exploration_decay=0.9995,
        min_exploration_rate=0.1
    )
    
    num_episodes = 15000
    print_interval = 500  # Less frequent printing for speed
    success_count = 0
    midpoint_reached_count = 0
    returned_to_base_count = 0
    
    print("Starting fast training...")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    
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
                
                current_pos = env.current_pos
                step_ratio = steps / env.max_steps
                
                dist_to_midpoint = np.linalg.norm(current_pos - midpoint)
                closest_to_midpoint = min(closest_to_midpoint, dist_to_midpoint)
                
                dist_to_base = np.linalg.norm(current_pos - env.goal_pos)
                max_distance_from_base = max(max_distance_from_base, dist_to_base)
                
                # Communication quality with strong penalties
                comm_quality = info['communication_quality']
                shaped_reward = comm_quality * 2.0
                
                # STRONG penalties for being in bad communication zones
                if list(current_pos) in env.nlos_both:
                    shaped_reward -= 5.0
                elif list(current_pos) in env.nlos_single:
                    shaped_reward -= 2.0
                
                # Bonus for good communication
                if comm_quality > 0.7:
                    shaped_reward += 3.0
                
                # Phase-based rewards
                if step_ratio < 0.5:
                    # EARLY PHASE: go to midpoint
                    if dist_to_base < 3.0:
                        shaped_reward -= 1.5
                    
                    shaped_reward += dist_to_base * 1.0
                    
                    if dist_to_midpoint < 5.0:
                        shaped_reward += (5.0 - dist_to_midpoint) * 4.0
                    
                    if dist_to_midpoint < 2.5:
                        shaped_reward += 40.0
                        reached_midpoint = True
                        
                else:
                    # LATE PHASE: return to base
                    if dist_to_base > 5.0:
                        shaped_reward -= 2.0
                    
                    shaped_reward += (15.0 - dist_to_base) * 1.5
                    
                    if reached_midpoint:
                        shaped_reward += 3.0
                        if dist_to_base < 3.0:
                            shaped_reward += 12.0
                
                if info['hit_obstacle']:
                    shaped_reward -= 10.0
                
                agent.update(state, action, shaped_reward, next_state, terminated)
                
                state = next_state
                total_reward += shaped_reward
                steps += 1
                episode_comm_quality += info['communication_quality']
            
            # End of episode evaluation
            final_pos = env.current_pos
            dist_to_base_final = np.linalg.norm(final_pos - env.goal_pos)
            
            reached_near_midpoint = closest_to_midpoint < 3.0
            returned_to_base = dist_to_base_final < 2.0
            explored_away = max_distance_from_base > 8.0
            
            # End bonuses
            if explored_away:
                total_reward += 30.0
                
            if reached_near_midpoint:
                midpoint_reached_count += 1
                total_reward += 50.0
                
            if returned_to_base:
                returned_to_base_count += 1
                total_reward += 40.0
            
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
            
            agent.decay_epsilon()
            
            # Print progress (less frequently)
            if (episode + 1) % print_interval == 0:
                elapsed_time = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed_time
                eta_seconds = (num_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                avg_reward = np.mean(agent.episode_rewards[-print_interval:])
                avg_length = np.mean(agent.episode_lengths[-print_interval:])
                avg_comm = np.mean(agent.communication_qualities[-print_interval:])
                success_rate = (success_count / (episode + 1)) * 100
                midpoint_rate = (midpoint_reached_count / (episode + 1)) * 100
                return_rate = (returned_to_base_count / (episode + 1)) * 100
                
                # Calculate recent performance (last 500 episodes)
                recent_success = success_count - agent.successful_episodes[max(0, episode - print_interval)]
                recent_midpoint = midpoint_reached_count - agent.reached_midpoint_count[max(0, episode - print_interval)]
                recent_return = returned_to_base_count - agent.returned_to_base_count[max(0, episode - print_interval)]
                
                print(f"Episode {episode + 1}/{num_episodes} | {eps_per_sec:.1f} eps/sec | ETA: {eta_minutes:.1f} min")
                print(f"  Avg Reward: {avg_reward:.1f} | Avg Steps: {avg_length:.1f} | Epsilon: {agent.epsilon:.3f}")
                print(f"  Avg Comm Quality: {avg_comm:.3f} {'✓ GOOD' if avg_comm > 0.6 else '✗ POOR'}")
                print(f"  Overall Rates - Success: {success_rate:.1f}% | Midpoint: {midpoint_rate:.1f}% | Return: {return_rate:.1f}%")
                print(f"  Last {print_interval} - Success: {recent_success} | Midpoint: {recent_midpoint} | Return: {recent_return}")
                print()
            
            # Quick progress indicator every 100 episodes
            if (episode + 1) % 100 == 0 and (episode + 1) % print_interval != 0:
                print(f"  {episode + 1}/{num_episodes}...", end='\r')
        
        print("\n")  # Clear the progress line
        
        total_time = time.time() - start_time
        
        # Save results
        agent.save_q_table('trained_q_table.npy')
        agent.save_training_stats('training_stats.json')
        
        # Plot training results
        print("Generating training plots...")
        plot_training_stats(
            agent.episode_rewards, 
            agent.episode_lengths, 
            agent.successful_episodes,
            agent.communication_qualities
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print(f"Total Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"Speed: {num_episodes/total_time:.1f} episodes/second")
        print(f"Final Success Rate: {(success_count / num_episodes) * 100:.1f}%")
        print(f"Final Midpoint Rate: {(midpoint_reached_count / num_episodes) * 100:.1f}%")
        print(f"Final Return Rate: {(returned_to_base_count / num_episodes) * 100:.1f}%")
        print(f"Final Comm Quality: {np.mean(agent.communication_qualities[-1000:]):.3f}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        total_time = time.time() - start_time
        print(f"Completed {episode + 1} episodes in {total_time/60:.1f} minutes")
        agent.save_q_table('trained_q_table_interrupted.npy')
        agent.save_training_stats('training_stats_interrupted.json')
    finally:
        env.close()
    
    return agent

if __name__ == "__main__":
    agent = train_q_learning_fast()
    
    # Optionally test the agent after training
    print("\nTraining complete! Run test.py to see the trained agent in action.")
