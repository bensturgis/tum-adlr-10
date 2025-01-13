import gymnasium as gym
import numpy as np
from typing import Dict

from metrics.evaluation_metric import EvaluationMetric

class MultiStepPredictiveAccuracyEvaluator(EvaluationMetric):
    """
    Evaluates the multi-step predictive accuracy of a learned environment against a true environment.
    """
    def __init__(
        self, true_env: gym.Env, learned_env: gym.Env, state_bounds: Dict[str, float], 
        num_trajectories: int, trajectory_horizon: int, num_initial_states: int,
        num_prediction_steps: int
    ) -> None:
        """
        Initializes the multi-step evaluator with both true and learned environments,
        and prepares a set of trajectories from the true environment for evaluation.

        Args:
            true_env (gym.Env): The true environment used to generate trajectories.
            learned_env (gym.Env): The learned environment whose predictions are to be evaluated.
            state_bounds (Dict[str, float]): State bounds a dynamical system can reach within a given 
                                             horizon for sampling the start states of the trajectories.
            num_trajectories (N_2) (int): Number of full trajectories to roll out in the true environment.
            trajectory_horizon (int): Length of the full trajectories.
            num_initial_states (N_3) (int): Number of initial states to sample from each trajectory
                                            for multi-step evaluation.
            num_prediction_steps (M) (int): Number of steps to predict ahead in the learned environment.
        """
        self.learned_env = learned_env
        self.num_trajectories = num_trajectories
        self.num_prediction_steps = num_prediction_steps
        self.trajectory_horizon = trajectory_horizon
        self.num_initial_states = num_initial_states
        self.name = "Multi Step Predictive Accuracy"

        # A list to store data for each trajectory, including states, actions, and
        # the chosen initial indices from which we'll start multi-step rollouts
        self.trajectories_data = []

        # Use scaled bounds for sampling start states of the trajectories
        state_low = 0.5 * np.array([state_bounds["min_position"], state_bounds["min_velocity"]])
        state_high = 0.5 * np.array([state_bounds["max_position"], state_bounds["max_velocity"]])

        # Sample start states uniformly within the computed range
        start_states = np.random.uniform(
            low=state_low,
            high=state_high,
            size=(self.num_trajectories, true_env.observation_space.shape[0])
        )

        # Generate trajectories by rolling out in the true environment starting from
        # a start state
        for start_state in start_states:
            # Set the true environment to a start state
            true_env.state = start_state

            # Collect states and actions
            states = [start_state]
            actions = []

            for _ in range(trajectory_horizon):
                # Sample a random action
                sampled_action = true_env.action_space.sample()[0]

                # Step in the true environment
                next_state, _, terminated, truncated, _ = true_env.step(sampled_action)

                # If the episode ends, stop collecting
                if terminated or truncated:
                    break

                actions.append(sampled_action)
                states.append(next_state)

            states_array = np.array(states)
            actions_array = np.array(actions)
            trajectory_length = states_array.shape[0]

            # Trajectories shorter than the number of prediction steps cannot be used for evaluation
            if trajectory_length <= num_prediction_steps:
                continue

            # Select valid starting indices from which we can predict num_prediction_steps ahead
            valid_indices = np.arange(trajectory_length - num_prediction_steps)

            # Randomly sample the desired number of initial states (or all if fewer remain)
            chosen_indices = np.random.choice(
                valid_indices,
                size=min(num_initial_states, len(valid_indices)),
                replace=False
            )

            # Store the data for this trajectory
            self.trajectories_data.append({
                "states": states_array,
                "actions": actions_array,
                "initial_indices": chosen_indices
            })

    def evaluate(self) -> float:
        """
        Runs multi-step rollouts in the learned environment from sampled initial states
        and computes the root mean squared error (RMSE) between the learned environment's
        predicted states and the true environment's states.

        Returns:
            float: The multi-step predictive RMSE for the learned environment.
        """
        squared_errors = []

        # Iterate over all collected trajectories
        for traj_dict in self.trajectories_data:
            states = traj_dict["states"]
            actions = traj_dict["actions"]
            initial_indices = traj_dict["initial_indices"]

            # For each sampled initial state, run num_prediction_steps in the learned env
            for idx in initial_indices:
                # Set the learned environment's state to the sampled initial state
                self.learned_env.state = states[idx].copy()

                # Flag to detect early termination
                early_termination = False

                # Step forward num_prediction_steps in the learned environment
                for step_ahead in range(self.num_prediction_steps):
                    action = actions[idx + step_ahead]
                    _, _, terminated, truncated, _ = self.learned_env.step(action)

                    if terminated or truncated:
                        early_termination = True
                        break

                # If rollout did not terminate early, compute the prediction error
                if not early_termination:
                    predicted_state = self.learned_env.state
                    true_state = states[idx + self.num_prediction_steps]
                    error = np.sum((predicted_state - true_state) ** 2)
                    squared_errors.append(error)

        rmse = np.sqrt(np.mean(squared_errors))
        return rmse

    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "num_trajectories": self.num_trajectories,
            "trajectory_horizon": self.trajectory_horizon,
            "num_initial_states": self.num_initial_states,
            "num_prediction_steps": self.num_prediction_steps,
        }
        return parameter_dict
