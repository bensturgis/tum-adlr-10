import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset

from sampling_methods.sampling_method import SamplingMethod

class RandomExploration(SamplingMethod):
    def __init__(self, horizon: int) -> None:
        """
        Initialize RandomExploration object.

        Args:
            horizon (T) (int): The total planning horizon (T).
        """
        super().__init__(horizon=horizon)

    def sample(self, true_env: gym.Env, learned_env: gym.Env = None) -> TensorDataset:
        """
        Perform random exploration in the environment and collect transitions.
        
        Args:
            true_env (gym.Env): The true environment for the actual data collection.
            learned_env (gym.Env): Ignored in this method. Included for consistency
                                   with the SamplingMethod interface.

        Returns:
            TensorDataset: Collected (state, action, next state) pairs.
        """
        # Lists to store states, actions, and next states collected during exploration
        states = []
        actions = []
        next_states = []
        
        # Reset the environment to initial state
        _, _ = true_env.reset()
        
        # Perform exploration for up to 'horizon' steps
        for _ in range(self.horizon):    
            # Append the current state to the list
            states.append(true_env.unwrapped.state)
            
            # Sample a random action from the environment's action space
            action = true_env.action_space.sample()
            
            # Step the environment using the sampled action
            next_state, _, terminated, truncated, _ = true_env.step(action)
            
            # Ensure the action has the correct dimensionality (expand if scalar)
            action = np.expand_dims(action, axis=0) if action.ndim < 1 else action
            
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
