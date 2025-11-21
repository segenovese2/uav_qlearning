# test_agent.py
import time
import numpy as np
import gymnasium as gym
from environments.uav_env import UAVEnv
import os
import pygame
import json

# Configuration - choose agent by string
# Options: "QLEARNING", "DQN", "PPO", "SAC", "A2C"
AGENT = "PPO"
MODEL_PATH = "trained_agent.pt"  # For neural network agents (DQN, PPO, SAC, A2C)
Q_TABLE_PATH = "trained_q_table.npy"  # For Q-learning Q-table
SLEEP = 0.25  # seconds between steps for slow visualisation

def load_q_agent(env):
    from agents.q_learning_agent import QLearningAgent
    # instantiate with shapes from env
    sample_obs, _ = env.reset()
    agent = QLearningAgent(state_size=len(sample_obs), action_size=env.action_space.n)
    # load Q-table
    if os.path.exists(Q_TABLE_PATH):
        agent.load(Q_TABLE_PATH)
        print(f"Loaded Q-table from {Q_TABLE_PATH}")
    else:
        raise FileNotFoundError(f"{Q_TABLE_PATH} not found for Q-learning agent.")
    return agent

def load_dqn_agent(env):
    from agents.dqn_agent import DQNAgent

    sample_obs, _ = env.reset()
    agent = DQNAgent(state_size=len(sample_obs), action_size=env.action_space.n)
    
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded DQN model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found for DQN agent.")
    return agent

def load_ppo_agent(env):
    from agents.PPO_agent import PPOAgent

    sample_obs, _ = env.reset()
    agent = PPOAgent(state_size=len(sample_obs), action_size=env.action_space.n)
    
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded PPO model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found for PPO agent.")
    return agent

def load_sac_agent(env):
    from agents.SAC_agent import SACAgent

    sample_obs, _ = env.reset()
    agent = SACAgent(state_size=len(sample_obs), action_size=env.action_space.n)
    
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded SAC model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found for SAC agent.")
    return agent

def load_a2c_agent(env):
    from agents.A2C_agent import A2CAgent

    sample_obs, _ = env.reset()
    agent = A2CAgent(state_size=len(sample_obs), action_size=env.action_space.n)
    
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded A2C model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found for A2C agent.")
    return agent

def choose_action(agent, state, agent_type="DQN"):
    """
    Choose action based on agent type.
    All agents are tested in inference mode (no exploration).
    """
    if agent_type.upper() in ["QLEARNING", "DQN", "PPO", "SAC", "A2C"]:
        # All our custom agents have choose_action(state, training=False)
        return agent.choose_action(state, training=False)
    else:
        raise RuntimeError(f"Unknown agent type: {agent_type}")

def run_test():
    env = UAVEnv(grid_size=15, render_mode="human")
    
    # Load appropriate agent
    if AGENT.upper() == "QLEARNING":
        agent = load_q_agent(env)
    elif AGENT.upper() == "DQN":
        agent = load_dqn_agent(env)
    elif AGENT.upper() == "PPO":
        agent = load_ppo_agent(env)
    elif AGENT.upper() == "SAC":
        agent = load_sac_agent(env)
    elif AGENT.upper() == "A2C":
        agent = load_a2c_agent(env)
    else:
        raise ValueError("Unknown AGENT. Choose 'QLEARNING', 'DQN', 'PPO', 'SAC', or 'A2C'.")

    state, _ = env.reset()
    done = False
    trajectory = [env.current_pos.copy()]
    total_reward = 0
    steps = 0

    try:
        while True:
            action = choose_action(agent, state, agent_type=AGENT)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            trajectory.append(env.current_pos.copy())
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(SLEEP)

            if terminated or truncated:
                print(f"\nEpisode finished!")
                print(f"Total steps: {steps}")
                print(f"Total reward: {total_reward:.2f}")
                break
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.close()
        np.save("trajectory.npy", np.array(trajectory))
        print(f"Saved trajectory.npy ({len(trajectory)} positions)")
        print(f"Agent: {AGENT} | Steps: {steps} | Reward: {total_reward:.2f}")

if __name__ == "__main__":
    run_test()
