import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple

def create_dataloader(dataset: TensorDataset, batch_size: int) -> DataLoader:
    """
    Creates a PyTorch DataLoader from the given dataset with the specified batch size.

    Args:
        dataset (Dataset): A PyTorch Dataset instance containing samples to be batched.
        batch_size (int): The number of samples per batch to load.

    Returns:
        DataLoader: A DataLoader instance that provides an iterable over the dataset,
                    returning batches of size `batch_size`.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def combine_datasets(dataset1: TensorDataset, dataset2: TensorDataset) -> TensorDataset:
    """
    Combines two PyTorch TensorDatasets by concatenating their tensors along the first (batch) dimension.

    Args:
        dataset1 (TensorDataset): The first dataset to combine.
        dataset2 (TensorDataset): The second dataset to combine.
    
    Returns:
        TensorDataset: A new dataset containing all samples from both `dataset1` and `dataset2`.
                       The concatenation occurs along dimension 0 for each corresponding tensor
                       in the datasets.
    """
    tensors1 = dataset1.tensors
    tensors2 = dataset2.tensors
    combined_tensors = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(tensors1, tensors2)]
    return TensorDataset(*combined_tensors)

def compute_state_bounds(env: gym.Env, horizon: int) -> Dict[int, np.array]:
    """
    Computes the minimum and maximum values for each state dimension over a given horizon
    by applying constant minimum and maximum actions.

    Args:
        env (gym.Env): The environment instance.
        horizon (int): Number of simulation steps to determine bounds.

    Returns:
        Dict[int, np.ndarray]: A dictionary mapping each state dimension index to a NumPy array
                                containing its [min, max] values.
    """
    def simulate(action: np.ndarray, dim_idx: int) -> Tuple[float, float]:
        """
        Simulates the environment with a constant action and tracks min/max for a specific dimension.
        """ 
        env.reset()  # Reset to initial state
        min_value = env.state[dim_idx]
        max_value = env.state[dim_idx]

        for _ in range(horizon):
            next_state, _, _, _, _ = env.step(action)  # Apply action
            # Update bounds
            min_value = min(min_value, next_state[dim_idx])
            max_value = max(max_value, next_state[dim_idx])
        
        return min_value, max_value

    min_action = env.action_space.low  # Minimum action values
    max_action = env.action_space.high  # Maximum action values

    # Initialize bounds with infinities
    state_bounds = {}
    
    for dim_idx in env.state_dim_names.keys():
        # Simulate with min and max actions
        min_val_min_act, max_val_min_act = simulate(min_action, dim_idx)
        min_val_max_act, max_val_max_act = simulate(max_action, dim_idx)
        
        # Determine overall min and max
        overall_min_val = min(min_val_min_act, min_val_max_act)
        overall_max_val = max(max_val_min_act, max_val_max_act)
        
        state_bounds[dim_idx] = np.array([overall_min_val, overall_max_val], dtype=np.float32)

    return state_bounds

def create_test_dataset(
    true_env: gym.Env, sampling_bounds: Dict[int, np.array], num_samples: int
) -> TensorDataset:
    """
    Samples states and actions from a true environment, computes their next states,
    and returns a TensorDataset containing (state, action, next_state) triples.

    Args:
        true_env (gym.Env): The true environment.
        sampling_bounds (Dict[int, np.array]): Sampling bounds for each state dimension, 
            specifying [min, max] values.
        num_samples (int): Number of samples to generate.

    Returns:
        TensorDataset: A PyTorch dataset containing (state, action, next_state) triples
                       for valid transitions.
    """
    # Sample states
    sampled_states = true_env.sample_states(num_samples, sampling_bounds)
    
    # Sample actions from the environment's action space
    sampled_actions = np.array([true_env.action_space.sample() for _ in range(num_samples)])
    
    valid_states = []
    valid_actions = []
    valid_next_states = []
    
    # Compute next states from the true environment
    for i in range(num_samples):
        state = sampled_states[i]
        action = sampled_actions[i]

        # Set the environment state
        true_env.state = state
        next_state, _, terminated, truncated, _ = true_env.step(action)

        # Keep only valid transitions (no termination/truncation)
        if not terminated and not truncated:
            valid_states.append(state)
            valid_actions.append(action)
            valid_next_states.append(next_state)

    # Convert valid transitions to Tensors and create a dataset
    states_tensor = torch.from_numpy(np.array(valid_states)).float()
    actions_tensor = torch.from_numpy(np.array(valid_actions)).float()
    next_states_tensor = torch.from_numpy(np.array(valid_next_states)).float()

    dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor)
    return dataset