import numpy as np
import matplotlib.pyplot as plt
from env import BanditEnv

def plot_bandit_reward_distributions(env: BanditEnv, num_pulls: int = 10_000):
    """Plot the reward distributions of the bandit arms

    Args:
        env (BanditEnv): bandit environment
        num_pulls (int): number of times to pull each arm
    """
    plt.figure(figsize=(15, 15))
    rewards = np.zeros((env.k, num_pulls))
    for i in range(env.k):
        for j in range(num_pulls):
            rewards[i, j] = env.step(i)
    
    # Set up the violin plot
    parts = plt.violinplot(rewards.T, showmeans=False, showextrema=False, showmedians=False)


    # Adding horizontal lines for means
    means = np.mean(rewards, axis=1)
    
    # Connecting mean values with solid lines and adding custom labels
    for i in range(env.k):
        plt.hlines(means[i], i + 0.75, i + 1.25, color='black', linestyle='solid', alpha=0.5)
        plt.text(i + 1.6, means[i], f'$q_*({i+1})$', color='black', ha='right', va='center')

    # Label axes and set title
    plt.ylabel('Reward distribution', )
    plt.xlabel('Action')

    # Draw a horizontal dotted line at y=0
    plt.axhline(0, color='grey', linestyle='dotted')

    # hide top left and right spines from plot
    spines = [plt.gca().spines[spine] for spine in ['top', 'right', 'left']]
    for spine in spines:
        spine.set_visible(False)

    # Set x-ticks to match the number of arms and label them
    plt.xticks(range(1, env.k + 1), [f'{i+1}' for i in range(env.k)])

    # Set y-ticks to range from -3 to 3 in increments of 1
    plt.yticks(np.arange(-3, 4, 1))

    plt.title(f'Reward Distributions of Bandit Arms (Number of Pulls = {num_pulls:,})')
