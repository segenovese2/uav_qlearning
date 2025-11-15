import gymnasium as gym
import numpy as np
import pygame
import json

from utils.plotter import plot_training_stats
from environments.uav_env import UAVEnv

# ============================
# SELECT AGENT HERE
# ============================
from agents.q_learning_agent import QLearningAgent as Agent
#from agents.dqn_agent import DQNAgent as Agent
# ============================


def train(agent_cls, num_episodes=15000, show_training=True):
    render_mode = "human" if show_training else None
    env = UAVEnv(grid_size=15, render_mode=render_mode)

    state, _ = env.reset()

    # Create agent
    agent = agent_cls(
        state_size=len(state),
        action_size=env.action_space.n
    )

    # Storage for training stats
    episode_rewards = []
    epsilons = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.update(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward

            if show_training:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return agent
                    
        # Epsilon decal (Q-learning)
        if hasattr(agent, "decay"):
            agent.decay()

        # Store stats
        episode_rewards.append(total_reward)
        epsilons.append(getattr(agent, "epsilon", None))

        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Reward={total_reward:.2f} | ε={agent.epsilon:.3f}")

    env.close()

    # Save training stats
    stats = {
        "episode_rewards": episode_rewards,
        "epsilons": epsilons
    }

    with open("training_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    np.save("training_rewards.npy", np.array(episode_rewards))

    print("\nSaved training_stats.json and training_rewards.npy")

    # Save trained model
    model_path = "trained_q_table.npy"

    if hasattr(agent, "save"):
        agent.save(model_path)
        print(f"Saved model -> {model_path}")
    else:
        raise RuntimeError("Agent has no save(filepath) method")
    
    return agent

if __name__ == "__main__":
    train(Agent, show_training=True)
