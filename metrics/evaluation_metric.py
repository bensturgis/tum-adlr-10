from abc import ABC, abstractmethod
from typing import Dict

class EvaluationMetric(ABC):
    """
    Abstract base class defining the interface for an evaluation metric.
    """

    def __init__(self) -> None:
        """
        Initializes the evaluation metric.
        """
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """
        Computes the root mean squared error (RMSE) between the learned environment's
        predicted states and the true environment's states.

        Returns:
            float: The RMSE for the learned environment.
        """
        pass

    @abstractmethod
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        pass