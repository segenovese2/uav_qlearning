# [file name]: uav_qlearning/test_qnet.py
import gymnasium as gym
import numpy as np
import pygame
from environments.uav_env import UAVEnv
from agents.qnet_agent import QNetAgent

def test_qnet_agent():
    """Test the trained Q-Network agent with visualization"""
    env = UAVEnv(grid_size=15, render_mode="human")
    
    state_size = 5
    action_size = env.action_space.n
    
    # Create agent and load trained model
    agent = QNetAgent(state_size, action_size)
    try:
        agent.load_model('trained_qnet_model.pth')
    except FileNotFoundError:
        print("No trained Q-Network model found. Please run train_qnet.py first.")
        return
    
    agent.epsilon = 0.0  # No exploration during testing
    
    print("Testing trained Q-Network agent...")
    print("Close the window or press Ctrl+C to stop\n")
    
    num_test_episodes = 10
    success_count = 0
    midpoint_count = 0
    return_count = 0
    
    try:
        for episode in range(num_test_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            episode_comm_quality = 0
            
            closest_to_midpoint = float('inf')
            midpoint = np.mean(env.users, axis=0)
            
            print(f"\n=== Test Episode {episode + 1}/{num_test_episodes} ===")
            env.render()
            
            while not (terminated or truncated):
                # Choose best action (no exploration)
                action = agent.choose_action(state, training=False)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Track metrics
                dist_to_midpoint = np.linalg.norm(env.current_pos - midpoint)
                closest_to_midpoint = min(closest_to_midpoint, dist_to_midpoint)
                
                state = next_state
                total_reward += reward
                steps += 1
                episode_comm_quality += info['communication_quality']
                
                env.render()
                
                # Check for quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                
                # Add small delay for visibility
                pygame.time.wait(50)
            
            # Evaluate episode
            avg_comm_quality = episode_comm_quality / steps if steps > 0 else 0
            final_pos = env.current_pos
            dist_to_base = np.linalg.norm(final_pos - env.goal_pos)
            
            reached_midpoint = closest_to_midpoint < 3.0
            returned_to_base = dist_to_base < 2.0
            good_comm = avg_comm_quality > 0.6
            
            success = reached_midpoint and returned_to_base
            
            if reached_midpoint:
                midpoint_count += 1
            if returned_to_base:
                return_count += 1
            if success:
                success_count += 1
            
            # Print results
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status}")
            print(f"  Steps: {steps}/50")
            print(f"  Final Position: {final_pos}")
            print(f"  Distance to Base: {dist_to_base:.2f}")
            print(f"  Closest to Midpoint: {closest_to_midpoint:.2f}")
            print(f"  Avg Comm Quality: {avg_comm_quality:.3f} {'(Good)' if good_comm else '(Poor)'}")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Reached Midpoint: {'Yes' if reached_midpoint else 'No'}")
            print(f"  Returned to Base: {'Yes' if returned_to_base else 'No'}")
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        print(f"Complete Success Rate: {(success_count/num_test_episodes)*100:.1f}% ({success_count}/{num_test_episodes})")
        print(f"Reached Midpoint: {(midpoint_count/num_test_episodes)*100:.1f}% ({midpoint_count}/{num_test_episodes})")
        print(f"Returned to Base: {(return_count/num_test_episodes)*100:.1f}% ({return_count}/{num_test_episodes})")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    test_qnet_agent()
