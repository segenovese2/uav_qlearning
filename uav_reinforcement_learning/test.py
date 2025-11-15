# test_agent.py
import time
import numpy as np
import gymnasium as gym
from environments.uav_env import UAVEnv
import os

# Configuration - choose agent by string
# Options: "QLEARNING", "DQN"
AGENT = "QLEANING"
MODEL_PATH = "trained_q_table.npy"  # Q-learning Q-table saved at root as you said
SLEEP = 0.25  # seconds between steps for slow visualisation

def load_q_agent(env):
    from agents.q_learning_agent import QLearningAgent
    # instantiate with shapes from env
    sample_obs, _ = env.reset()
    agent = QLearningAgent(state_size=len(sample_obs), action_size=env.action_space.n)
    # load Q-table
    if os.path.exists(MODEL_PATH):
        agent.load_q_table(MODEL_PATH)
        print(f"Loaded Q-table from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found for Q-learning agent.")
    return agent

def load_dqn_agent(env):
    # Try to import a DQN agent class; user will have to provide this module/class
    try:
        from agents.dqn_agent import DQNAgent
    except Exception as e:
        raise ImportError("DQNAgent not found in agents/dqn_agent.py") from e

    sample_obs, _ = env.reset()
    agent = DQNAgent(state_size=len(sample_obs), action_size=env.action_space.n)
    # assume agent has load() method that accepts MODEL_PATH or uses its own path
    if hasattr(agent, "load"):
        # try loading the same filename — if your DQN uses another file, change MODEL_PATH.
        if os.path.exists(MODEL_PATH):
            agent.load(MODEL_PATH)
            print(f"Loaded DQN model from {MODEL_PATH}")
        else:
            print(f"{MODEL_PATH} not found; calling agent.load() without path (agent may handle its own path).")
            agent.load()
    else:
        raise AttributeError("DQNAgent class needs a load() method.")
    return agent

def choose_action(agent, state):
    # For Q-learning agent
    if hasattr(agent, "choose_action"):
        return agent.choose_action(state, training=False)
    # For DQN-like agents, attempt predict / act / policy methods
    if hasattr(agent, "predict"):
        # stable-baselines style
        action, _ = agent.predict(state, deterministic=True)
        return int(action)
    if hasattr(agent, "act"):
        return int(agent.act(state))
    raise RuntimeError("Agent does not expose a known policy method (choose_action / predict / act).")

def run_test():
    env = UAVEnv(grid_size=15, render_mode="human")
    if AGENT.upper() == "QLEARNING":
        agent = load_q_agent(env)
    elif AGENT.upper() == "DQN":
        agent = load_dqn_agent(env)
    else:
        raise ValueError("Unknown AGENT. Choose 'QLEARNING' or 'DQN'.")

    state, _ = env.reset()
    done = False
    trajectory = [env.current_pos.copy()]

    try:
        while True:
            action = choose_action(agent, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            trajectory.append(env.current_pos.copy())
            env.render()
            time.sleep(SLEEP)

            if terminated or truncated:
                print("Episode finished.")
                break
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        env.close()
        np.save("trajectory.npy", np.array(trajectory))
        print("Saved trajectory.npy (path taken).")

if __name__ == "__main__":
    run_test()
