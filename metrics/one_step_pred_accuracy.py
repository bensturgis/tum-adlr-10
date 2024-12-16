import gymnasium as gym
import numpy as np
from torch.utils.data import TensorDataset

class OneStepPredictiveAccuracyEvaluator:
    """
    Evaluates the one-step predictive accuracy of a learned environment against a true environment,
    using a given test dataset of (state, action, next_state) samples.
    """

    def __init__(self, learned_env: gym.Env, dataset: TensorDataset) -> None:
        """
        Initializes the evaluator with a learned environment and a pre-created dataset of transitions.

        Args:
            learned_env (gym.Env): The learned environment.
            dataset (TensorDataset): A dataset of (state, action, next_state) samples.
        """
        self.learned_env = learned_env
        self.dataset = dataset

    def compute_one_step_pred_accuracy(self) -> float:
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
            self.learned_env.state = state
            pred_next_state, _, pred_terminated, pred_truncated, _ = self.learned_env.step(action)

            # Skip if termination or truncation occurs
            if pred_terminated or pred_truncated:
                continue

            # Compute squared error
            squared_error = np.sum((pred_next_state - true_next_state) ** 2)
            squared_errors.append(squared_error)

        # Compute RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        return rmse