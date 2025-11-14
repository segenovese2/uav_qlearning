# [file name]: uav_qlearning/test.py
# [file content begin]
import gymnasium as gym
import numpy as np
import pygame
from environments.uav_env import UAVEnv

# ============================
# SELECT AGENT HERE
# ============================
from agents.q_learning_agent import QLearningAgent as agent
# from agents.dqn_agent import DQNAgent as Agent
# ============================

def test_trained_agent():
    # Create environment
    env = UAVEnv(grid_size=15, render_mode="human")
    
    # Get state and action sizes
    state_size = 5  # New state representation
    action_size = env.action_space.n
    
    # Create agent and load trained Q-table
    try:
        agent.load('trained_q_table.npy')
    except FileNotFoundError:
        print("No trained Q-table found. Please run train.py first.")
        return
    
    agent.epsilon = 0.0  # No exploration during testing
    
    print("Testing trained agent...")
    
    # Run multiple test episodes
    num_test_episodes = 10
    success_count = 0
    
    for episode in range(num_test_episodes):
        state, _ = env.reset()  # state is now a vector
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        episode_comm_quality = 0
        
        print(f"\n=== Test Episode {episode + 1} ===")
        env.render()
        
        while not (terminated or truncated):
            # Choose best action (no exploration)
            action = agent.choose_action(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            episode_comm_quality += info['communication_quality']
            
            env.render()
            
            # Check for Pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            if steps > 50:  # Safety break
                break
        
        # Calculate average communication quality
        avg_comm_quality = episode_comm_quality / steps if steps > 0 else 0
        
        # NEW SUCCESS CONDITION: Same as training
        reached_goal = np.array_equal(env.current_pos, env.goal_pos)
        good_communication = avg_comm_quality > 0.6
        moved_away_from_start = any(np.linalg.norm(pos - env.start_pos) > 5 for pos in env.trajectory)
        
        success = (reached_goal and steps <= env.max_steps) or (good_communication and moved_away_from_start and steps > 10)
        
        if success:
            success_count += 1
            print("✓ SUCCESS: Good performance!")
        else:
            print("✗ FAILED: Poor performance")
        
        print(f"Total reward: {total_reward}, Steps: {steps}, Avg Comm: {avg_comm_quality:.2f}")
    
    success_rate = (success_count / num_test_episodes) * 100
    print(f"\n=== Final Results ===")
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{num_test_episodes})")
    
    env.close()

if __name__ == "__main__":
    test_trained_agent()
# [file content end]
