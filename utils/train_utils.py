import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict

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

def create_test_dataset(
    true_env: gym.Env, state_bounds: Dict[str, float], num_samples: int
) -> TensorDataset:
    """
    Samples states and actions from a true environment, computes their next states,
    and returns a TensorDataset containing (state, action, next_state) triples.

    Args:
        true_env (gym.Env): The true environment.
        state_bounds (Dict[str, float]): State bounds a dynamical system can reach within
                                         a given horizon for state sampling.
        num_samples (int): Number of samples to generate.

    Returns:
        TensorDataset: A PyTorch dataset containing (state, action, next_state) triples
                       for valid transitions.
    """
    # Use scaled bounds for sampling states
    state_low = 0.5 * np.array([state_bounds["min_position"], state_bounds["min_velocity"]])
    state_high = 0.5 * np.array([state_bounds["max_position"], state_bounds["max_velocity"]])

    # Sample states uniformly within the computed range
    sampled_states = np.random.uniform(
        low=state_low,
        high=state_high,
        size=(num_samples, true_env.observation_space.shape[0])
    )
    
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