from abc import ABC, abstractmethod
import gym
from torch.utils.data import TensorDataset

class SamplingMethod(ABC):
    def __init__(self, horizon: int) -> None:
        """
        Initialize SamplingMethod object.

        Args:
            horizon (int): The number of steps to explore in a given environment.
        """
        self.horizon = horizon

    @abstractmethod
    def sample(self, true_env: gym.Env, learned_env: gym.Env) -> TensorDataset:
        """
        Perform a sampling operation to collect (state, action, next state)
        pairs in a given environment.

        Args:
            true_env (gym.Env): The true environment for the actual data collection.
            learned_env (gym.Env): The learned environment using a dynamics model to find the most
                                   informative action sequence.

        Returns:
            TensorDataset: Collected (state, action, next state) pairs.
        """
        pass