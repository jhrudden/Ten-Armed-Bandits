import numpy as np
from typing import Tuple, Optional


class BanditEnv:
    """Multi-armed bandit environment"""

    def __init__(self, k: int, random_state: Optional[float] = None) -> None:
        """__init__.

        Args:
            k (int): number of arms/bandits
            random_state (float): random seed
        """
        self.k = k
        
        if random_state is not None:
            np.random.seed(random_state)

        self.reset()

    def reset(self) -> None:
        """Resets the mean payout/reward of each arm.
        This function should be called at least once after __init__()
        """
        # Initialize means of each arm distributed according to standard normal
        self.means = np.random.normal(size=self.k)

    def step(self, action: int) -> Tuple[float, int]:
        """Take one step in env (pull one arm) and observe reward

        Args:
            action (int): index of arm to pull
        """
        reward = np.random.normal(loc=self.means[action])

        return reward
