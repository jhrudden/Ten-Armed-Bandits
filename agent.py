import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Sequence


def argmax(arr: Sequence[float]) -> int:
    """
    Argmax that breaks ties randomly (np.argmax only returns first index in case of ties, we don't like this)

    Args:
        arr: sequence of values
    
    Returns:
        index of maximum value
    """
    max_val = np.max(arr)
    max_indices = np.where(arr == max_val)[0]
    return np.random.choice(max_indices)

class BanditAgent(ABC):
    Q = None # Q-values #   
    N = None # Number of times each arm was pulled
    t = None # Number of time steps taken

    def __init__(self, k: int, init: int, step_size: float) -> None:
        """Abstract bandit agent class

        Implements common functions for both epsilon greedy and UCB

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            step_size (float): step size
        """
        self.k = k
        self.init = init
        self.step_size = step_size
        self.reset()

    def reset(self) -> None:
        """
        Initialize or reset Q-values, counts and time step
        """
        self.Q = self.init * np.ones(self.k, dtype=np.float32)
        self.N = np.zeros(self.k, dtype=int)
        self.t = 0

    @abstractmethod
    def choose_action(self) -> int:
        """Choose which arm to pull"""
        raise NotImplementedError

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        raise NotImplementedError


class EpsilonGreedy(BanditAgent):
    def __init__(
        self, k: int, init: int, epsilon: float, step_size: Optional[float] = None
    ) -> None:
        """Epsilon greedy bandit agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            epsilon (float): random action probability
            step_size (float or None): step size. If None, then it is equal to 1 / N_t (dynamic step size)
        """
        super().__init__(k, init, step_size)
        self.epsilon = epsilon

    def choose_action(self):
        """
        Choose which arm to pull.

        (1-epsilon) of the time, choose the arm with the highest Q-value (argmax(Q)) (break ties randomly).
        epsilon of the time, choose a random arm uniformly

        Returns:
            action (int): index of arm to pull
        """
        random_val = np.random.uniform()
        # explore (random action)
        if random_val < self.epsilon:
            return np.random.randint(self.k)
        
        # exploit (greedy action)
        return argmax(self.Q)


    def update(self, action: int, reward: float) -> None:
        """
        Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        self.t += 1

        self.N[action] += 1

        step_ceof = 1 / self.N[action]

        if self.step_size is not None:
            step_ceof = self.step_size
        
        self.Q[action] += step_ceof * (reward - self.Q[action])

    def __str__(self) -> str:
        return f"$\epsilon = {self.epsilon}$"


class UCB(BanditAgent):
    def __init__(self, k: int, init: int, c: float, step_size: float) -> None:
        """Epsilon greedy bandit agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            c (float): UCB constant that controls degree of exploration
            step_size (float): step size (use constant step size in case of UCB)
        """
        super().__init__(k, init, step_size)
        self.c = c

    def choose_action(self):
        """Choose which arm to pull

        Use UCB action selection. Be sure to consider the case when N_t = 0 and break ties randomly (use argmax() from above)
        """
        # TODO
        action = None
        return action

    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        self.t += 1

        # TODO update self.N

        # TODO update self.Q
