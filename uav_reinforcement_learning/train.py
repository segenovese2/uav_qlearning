import gymnasium as gym
import numpy as np
import pygame
import json

from environments.uav_env import UAVEnv

# ============================
# SELECT AGENT HERE
# ============================
from agents.q_learning_agent import QLearningAgent as Agent
#from agents.dqn_agent import DQNAgent as Agent
#from agents.PPO_agent import PPOAgent as Agent

# NOT IMPLEMENTED YET
#rom agents.SAC_agent import SACAgent as Agent
#from agents.A2C_agent import A2CAgent as Agent
# ============================


def train(agent_cls, num_episodes=20000, show_training=True):
    render_mode = "human" if show_training else None
    env = UAVEnv(grid_size=15, render_mode=render_mode)

    state, _ = env.reset()

    if agent_cls.__name__ == "QLearningAgent":
        agent = agent_cls(
            observation_space = env.observation_space,
            action_space = env.action_space
        )
    else:
        agent = agent_cls(
            state_size = env.observation_space.shape[0],
            action_size = env.action_space.n
        )

    # Storage for training stats
    episode_rewards = []
    epsilons = []

    for episode in range(num_episodes):
        state, _ = env.reset()

        if hasattr(agent, "reset"):
            agent.reset()

        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent_name = agent.__class__.__name__

            if agent_name == "PPOAgent":
                agent.store_experience(reward, terminated or truncated)
            elif agent_name == "SACAgent":
                agent.store_experience(state, action, reward, next_state, terminated or truncated)
                agent.update()
            elif agent_name == "A2CAgent":
                agent.update(state, action, reward, next_state, terminated or truncated)
            else:
                agent.update(state, action, reward, next_state, terminated or truncated)

            state = next_state
            total_reward += reward

            if show_training:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return agent

        if agent.__class__.__name__ == 'PPOAgent':
            agent.update()

        # Epsilon decay (Q-learning + DQN only)
        if hasattr(agent, "decay"):
            agent.decay()

        # Store stats
        episode_rewards.append(total_reward)
        epsilons.append(getattr(agent, "epsilon", None))

        if (episode+1) % 100 == 0:
            epsilon_str = f"| ε={agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
            print(f"Episode {episode+1}/{num_episodes} | Reward={total_reward:.2f} | {epsilon_str}")

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
    if agent.__class__.__name__ == "QLearningAgent":
        model_path = "trained_q_table.npy"
    else:
        model_path = "trained_agent.pt"

    if hasattr(agent, "save"):
        agent.save(model_path)
        print(f"Saved model -> {model_path}")
    else:
        raise RuntimeError("Agent has no save(filepath) method")
    
    return agent

if __name__ == "__main__":
    train(Agent, show_training=True)

