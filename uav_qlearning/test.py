import gymnasium as gym  # Changed from gym to gymnasium
import numpy as np
import pygame  # Added pygame import
from environments.uav_env import UAVEnv
from agents.q_learning_agent import QLearningAgent

def test_trained_agent():
    # Create environment
    env = UAVEnv(grid_size=10, render_mode="human")
    
    # Calculate state size for Q-table
    state_size = env.grid_size * env.grid_size
    action_size = env.action_space.n
    
    # Create agent and load trained Q-table
    agent = QLearningAgent(state_size, action_size)
    try:
        agent.load_q_table('trained_q_table.npy')
    except FileNotFoundError:
        print("No trained Q-table found. Please run train.py first.")
        return
    
    agent.epsilon = 0.0  # No exploration during testing
    
    print("Testing trained agent...")
    
    # Run multiple test episodes
    num_test_episodes = 10
    success_count = 0
    
    for episode in range(num_test_episodes):
        state, _ = env.reset()  # Gymnasium returns (state, info)
        state_idx = env.get_state_index(state)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        print(f"\n=== Test Episode {episode + 1} ===")
        env.render()
        
        while not (terminated or truncated):
            # Choose best action (no exploration)
            action = agent.choose_action(state_idx, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_idx = env.get_state_index(next_state)
            
            state_idx = next_state_idx
            total_reward += reward
            steps += 1
            
            env.render()
            
            # Check for Pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            if steps > 50:  # Safety break
                break
        
        # Check if goal was reached
        if np.array_equal(env.current_pos, env.goal_pos):
            success_count += 1
            print("✓ SUCCESS: Reached the goal!")
        else:
            print("✗ FAILED: Did not reach the goal")
        
        print(f"Total reward: {total_reward}, Steps: {steps}")
    
    success_rate = (success_count / num_test_episodes) * 100
    print(f"\n=== Final Results ===")
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{num_test_episodes})")
    
    env.close()

if __name__ == "__main__":
    test_trained_agent()
