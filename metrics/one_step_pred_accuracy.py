import gymnasium as gym
import numpy as np
from typing import Dict

from metrics.evaluation_metric import EvaluationMetric
from utils.train_utils import create_test_dataset

class OneStepPredictionErrorEvaluator(EvaluationMetric):
    """
    Evaluates the one-step predictive accuracy of a learned environment against a true environment.
    """

    def __init__(
            self, true_env: gym.Env, learned_env: gym.Env, num_samples: int, horizon: int
        ) -> None:
        """
        Initializes the one-step evaluator with a true environment, a learned environment, and parameters 
        to generate a test dataset of transitions.

        Args:
            true_env (gym.Env): The true environment used to generate the dataset.
            learned_env (gym.Env): The learned environment to be evaluated.
            num_samples (int): The number of samples ((state, action, next_state) pairs) to include 
                               in the test dataset.
            horizon (int): Number of simulation steps. Required for calculating state bounds.
        """
        self.learned_env = learned_env
        self.num_samples = num_samples
        self.name = "One-Step Prediction Error"
        self.dataset = create_test_dataset(
            true_env=true_env,
            num_samples=num_samples,
            horizon=horizon
        )

    def evaluate(self) -> float:
        """
        Computes the one-step predictive accuracy (RMSE) of the learned environment versus the
        true environment using the provided dataset.

        Returns:
            float: One-step predictive accuracy (RMSE).
        """
        squared_errors = []

        # Iterate over the dataset samples
        for i in range(len(self.dataset)):
            state, action, true_next_state = self.dataset[i]

            state = state.numpy()
            action = action.numpy()
            true_next_state = true_next_state.numpy()

            # Set the initial state in the learned environment
            self.learned_env.set_state(state)
            pred_next_state, pred_terminated, pred_truncated = self.learned_env.step_no_reward(action)

            # Skip if termination or truncation occurs
            if pred_terminated or pred_truncated:
                continue
            # Compute squared error
            squared_error = np.sum((pred_next_state - true_next_state) ** 2)
            squared_errors.append(squared_error) 

        # Compute RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        return rmse
    
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "num_samples": self.num_samples,
        }
        return parameter_dict