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

def plot_optimal_action(data: np.ndarray, agent_labels: Sequence[str], alpha: float = 0.2, z_val: float = 1.96):
    """
    Plot the percentage of optimal actions taken over time

    Args:
        data (np.ndarray): data from simulation
        agent_labels (Sequence[str]): labels for each agent
        alpha (float): transparency of the shaded region
        z_val (float): z-value for confidence interval
    """
    n_agents, n_trials, n_steps = data.shape

    assert n_agents == len(agent_labels), "Number of agents and labels must match"

    for a in range(n_agents):
        scaled_data = data[a] * 100
        optimal_action_percentage = np.mean(scaled_data, axis=0)
        std_per_step = np.std(scaled_data, axis=0)
        margin_of_error = z_val * std_per_step / np.sqrt(n_steps)
        plt.fill_between(
            range(n_steps),
            optimal_action_percentage - margin_of_error,
            optimal_action_percentage + margin_of_error,
            alpha=alpha,
        )
        plt.plot(optimal_action_percentage, label=agent_labels[a])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action Taken')

# TODO: what should the upper bound be?
def plot_average_reward(data: np.ndarray, agent_labels: Sequence[str], upper_bound: float, alpha: float = 0., z_val: float = 1.96):
    """
    Plot the average reward over time

    Args:
        data (np.ndarray): data from simulation
        agent_labels (Sequence[str]): labels for each agent
        upper_bound (float): upper bound associated with highest possible 
        alpha (float): transparency of the shaded region
        z_val (float): z-value for confidence interval
    """
    n_agents, n_trials, n_steps = data.shape

    assert n_agents == len(agent_labels), "Number of agents and labels must match"

    for a in range(n_agents):
        average_reward_per_step = np.mean(data[a], axis=0)
        std_per_step = np.std(data[a], axis=0)
        margin_of_error = z_val * std_per_step / np.sqrt(n_steps)
        plt.fill_between(
            range(n_steps),
            average_reward_per_step - margin_of_error,
            average_reward_per_step + margin_of_error,
            alpha=alpha,
        )
        plt.plot(average_reward_per_step, label=agent_labels[a])

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')