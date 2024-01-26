import numpy as np
from typing import Tuple, Optional


class BanditEnv:
    """Multi-armed bandit environment"""

    def __init__(self, k: int, random_state: Optional[float] = None, stationary: bool = True) -> None:
        """__init__.

        Args:
            k (int): number of arms/bandits
            random_state (float): random seed
            stationary (bool): whether the environment is stationary
        """
        self.k = k
        self.stationary = stationary

        self.reset()

    def reset(self, random_state: Optional[float] = None) -> None:
        """Resets the mean payout/reward of each arm.
        This function should be called at least once after __init__()
        """
        if random_state is not None:
            self.random_state = random_state
            np.random.seed(self.random_state)
        if self.stationary:
            # Initialize means of each arm distributed according to standard normal
            self.means = np.random.normal(size=self.k)
        else:
            # Initialize means of each arm to 0
            self.means = np.zeros(self.k)

    def step(self, action: int) -> Tuple[float, int]:
        """Take one step in env (pull one arm) and observe reward

        Args:
            action (int): index of arm to pull
        """
        reward = np.random.normal(loc=self.means[action])

        if not self.stationary:
            self.means += np.random.normal(size=self.k, scale=0.01)

        return reward
