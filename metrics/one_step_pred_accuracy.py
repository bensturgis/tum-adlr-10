import gymnasium as gym
import numpy as np

class OneStepPredictiveAccuracyEvaluator:
    """
    Evaluates the one-step predictive accuracy of a learned environment against a true environment,
    using a fixed set of sampled states and actions.

    Once initialized, the class pre-samples 'num_samples' states and actions. Subsequent calls to 
    'compute_one_step_pred_accuracy' will reuse the same samples, ensuring consistent conditions 
    for repeated evaluations.
    """

    def __init__(self, true_env: gym.Env, learned_env: gym.Env, num_samples: int) -> None:
        """
        Initializes the evaluator by pre-sampling states and actions.

        Args:
            true_env (gym.Env): The true environment from which to determine true next states.
            learned_env (gym.Env): The learned environment.
            num_samples (int): The number of samples ((state, action) pairs) to evaluate.
        """
        self.true_env = true_env
        self.learned_env = learned_env
        self.num_samples = num_samples

        # TODO: Use random exploration to get reasonable data samples instead of
        # hard coding lower and upper bound of state space
        # Pre-sample states within the observation space bounds
        self.sampled_states = np.random.uniform(
            low=-0.5,
            high=0.5,
            size=(num_samples, true_env.observation_space.shape[0])
        )

        # Pre-sample actions from the action space
        self.sampled_actions = np.array([
            true_env.action_space.sample() for _ in range(num_samples)
        ])

    def compute_one_step_pred_accuracy(self) -> float:
        """
        Computes the one-step predictive accuracy (RMSE) of the learned environment versus the
        true environment using the pre-sampled states and actions.

        Returns:
            float: One-step predictive accuracy (RMSE).
        """
        squared_errors = []

        for i in range(self.num_samples):
            state = self.sampled_states[i]
            action = self.sampled_actions[i]

            # Set the same initial state in both environments
            self.true_env.state = state
            self.learned_env.state = state

            # Step both environments
            true_next_state, _, true_terminated, true_truncated, _ = self.true_env.step(action)
            pred_next_state, _, pred_terminated, pred_truncated, _ = self.learned_env.step(action)

            # Skip if termination or truncation occurs
            if true_terminated or true_truncated or pred_terminated or pred_truncated:
                continue

            # Compute squared error for this sample
            squared_error = np.sum((pred_next_state - true_next_state) ** 2)
            squared_errors.append(squared_error)

        # Compute RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        return rmse