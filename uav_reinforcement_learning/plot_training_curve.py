# plot_training_curve.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_rewards():
    # Primary: training_stats.json (from your training script)
    if os.path.exists("training_stats.json"):
        with open("training_stats.json", "r") as f:
            stats = json.load(f)
        # which key likely contains the episodic sum-rate? try 'episode_rewards'
        if "episode_rewards" in stats:
            return np.array(stats["episode_rewards"], dtype=float)
        # fallback: maybe 'sum_rates' etc.
        for k in ["sum_rates", "rewards", "rewards_per_episode"]:
            if k in stats:
                return np.array(stats[k], dtype=float)

    # Second: training_rewards.npy (generic fallback)
    if os.path.exists("training_rewards.npy"):
        return np.load("training_rewards.npy")

    # Third: trained_q_table.npy can't give episode rewards. return empty
    print("No training rewards file found (tried training_stats.json and training_rewards.npy).")
    return np.array([])

def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def plot_curve(window=50):
    rewards = load_rewards()
    if rewards.size == 0:
        print("No rewards to plot.")
        return

    episodes = np.arange(1, len(rewards) + 1)
    ma = moving_average(rewards, window)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, alpha=0.25, label="Episode reward (raw)")
    plt.plot(episodes[window-1:], ma, linewidth=2.0, label=f"Moving avg (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Expected sum-rate (bits/s/Hz)")
    plt.title("Expected sum-rate over episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300)
    print("Saved training_curve.png")

if __name__ == "__main__":
    plot_curve(window=50)
