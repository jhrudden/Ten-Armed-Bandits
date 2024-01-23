import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Dict

from env import BanditEnv
from agent import BanditAgent

def plot_bandit_reward_distributions(env: BanditEnv, num_pulls: int = 10_000):
    """Plot the reward distributions of the bandit arms

    Args:
        env (BanditEnv): bandit environment
        num_pulls (int): number of times to pull each arm
    """
    plt.figure(figsize=(15, 10))
    rewards = np.zeros((env.k, num_pulls))
    for i in range(env.k):
        for j in range(num_pulls):
            rewards[i, j] = env.step(i)
    
    # Set up the violin plot
    parts = plt.violinplot(rewards.T, showmeans=False, showextrema=False, showmedians=False)

    # Change colors of violins
    for pc in parts['bodies']:
        pc.set_alpha(0.7)


    # Adding horizontal lines for means
    means = np.mean(rewards, axis=1)
    
    # Connecting mean values with solid lines and adding custom labels
    for i in range(env.k):
        plt.hlines(means[i], i + 0.75, i + 1.25, color='black', linestyle='solid', alpha=0.5)
        plt.text(i + 1.26, means[i], f'$q_*({i+1})$', color='black', ha='left', va='center')

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

    # Set y-ticks to range from -4 to 4 in increments of 1
    plt.yticks(np.arange(-4, 5, 1))

    plt.title(f'Reward Distributions of Bandit Arms (Number of Pulls = {num_pulls:,})')

def run_simulation(agents: Sequence[BanditAgent], env: BanditEnv, n_steps: int, n_trials: int) -> dict:
    """
    Run a simulation of the bandit environment with the given agents

    Args:
        agents (Sequence[BanditAgent]): bandit agents
        env (BanditEnv): bandit environment
        n_steps (int): number of steps to run
        n_trials (int): number of trials to run

    Returns:
        dict: dictionary containing the results of the simulation
    """

    data = {
        "took_optimal_action": np.zeros((len(agents), n_trials, n_steps)),
        "rewards": np.zeros((len(agents), n_trials, n_steps)),
    }

    for i, agent in enumerate(agents):
        for j in range(n_trials):
            agent.reset()
            env.reset()

            optimal_action = np.argmax(env.means)

            for t in range(n_steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)

                if action == optimal_action:
                    data["took_optimal_action"][i, j, t] = 1

                data["rewards"][i, j, t] = reward
    return data

def plot_optimal_action(data: np.ndarray, num_steps: int, agent_labels: Sequence[str]):
    """
    Plot the percentage of optimal actions taken over time

    Args:
        data (np.ndarray): data from simulation
        num_steps (int): number of steps to run
        agent_labels (Sequence[str]): labels for each agent
    """
    for a in range(len(agent_labels)):
        optimal_action_percentage = np.cumsum(np.mean(data[a], axis=0)) / np.arange(1, num_steps + 1)
        plt.plot(optimal_action_percentage, label=agent_labels[a])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action Taken')

def plot_average_reward(data: np.ndarray, num_steps: int, agent_labels: Sequence[str]):
    """
    Plot the average reward over time

    Args:
        axes (Axes): matplotlib axes object
        data (np.ndarray): data from simulation
        num_steps (int): number of steps to run
        agent_labels (Sequence[str]): labels for each agent
    """
    for a in range(len(agent_labels)):
        average_reward = np.mean(data[a], axis=0)
        plt.plot(average_reward, label=agent_labels[a])
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
