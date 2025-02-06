import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import torch
from torch.utils.data import TensorDataset
from typing import Dict

from sampling_methods.sampling_method import SamplingMethod

class SoftActorCritic(SamplingMethod):
    def __init__(self, horizon: int, total_timesteps: int) -> None:
        """
        Initialize SoftActorCritic object.

        Args:
            horizon (T) (int): The total planning horizon.
            total_timesteps (int): The total number of timesteps to train the Soft Actor-Critic 
                                   model on the learned environment.
        """
        super().__init__(horizon=horizon)
        self.total_timesteps = total_timesteps
        self.name = "Soft Actor Critic"
        

    def sample(self, true_env: gym.Env, learned_env: gym.Env) -> TensorDataset:
        """
        Implements Random Shooting Model Predictive Control (RS+MPC) as described in Algorithm 4
        from the paper "Actively learning dynamical systems using Bayesian neural networks."

        Args:
            true_env (gym.Env): The true environment for the actual data collection.
            learned_env (gym.Env): The environment with a learned dynamics model.

        Returns:
            TensorDataset: A PyTorch TensorDataset containing the collected
                           (state, action, next_state) transitions.
        """
        # Enforce a maximum episode to solve finite-horizon optimization problem
        learned_env = gym.wrappers.TimeLimit(learned_env, max_episode_steps=80)
        
        # Initialize a Soft Actor-Critic model for the learned environment
        model = SAC(
            policy="MlpPolicy", env=learned_env, learning_starts=10000, ent_coef="auto_0.1",
            target_update_interval=2, verbose=1
        )

        # Train the SAC model to find most informative policy for the learned environment
        model.learn(total_timesteps=self.total_timesteps, log_interval=4)

        # Lists to store states, actions, and next states collected during exploration
        states = []
        actions = []
        next_states = []

        # Reset the environment to initial state
        true_env.reset()

        # Perform exploration for up to 'horizon' steps
        for _ in range(self.horizon):
            # Append the current state to the list
            states.append(true_env.state)

            # Predict the most informative action using the trained SAC model
            # with a deterministic policy
            action, _ = model.predict(true_env.state, deterministic=True)

            # Step the environment using the action
            next_state, _, terminated, truncated, _ = true_env.step(action)

            # Append the action and the resulting next state to their respective lists
            actions.append(action)
            next_states.append(next_state)

            # Reset if the environment reaches a terminal or truncated state
            if terminated or truncated:
                true_env.reset()

        # Convert the collected lists into tensors
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        action_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
        next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        
        return TensorDataset(state_tensor, action_tensor, next_state_tensor)
    
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "horizon": self.horizon,
            "total_timesteps": self.total_timesteps,
        }
        return parameter_dict