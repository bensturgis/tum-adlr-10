import torch
from torch.utils.data import TensorDataset
import numpy as np

def random_exploration(env, horizon):
    """
    Perform random exploration in the environment and collect transitions.
    
    Args:
        env: The environment to explore. Must follow the OpenAI Gym API.
        horizon: The number of steps to explore in the environment.

    Returns:
        A TensorDataset containing the explored states, actions, and next states.
    """
    # Lists to store states, actions, and next states collected during exploration
    states = []
    actions = []
    next_states = []
    
    # Reset the environment to initial state
    _, _ = env.reset()
    
    # Perform exploration for up to 'horizon' steps
    for _ in range(horizon):    
        # Append the current state to the list
        states.append(env.unwrapped.state)
        
        # Sample a random action from the environment's action space
        action = env.action_space.sample()
        
        # Step the environment using the sampled action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Ensure the action has the correct dimensionality (expand if scalar)
        action = np.expand_dims(action, axis=0) if action.ndim < 1 else action
        
        # Append the action and the resulting next state to their respective lists
        actions.append(action)
        next_states.append(next_state)
        
        # Reset if the environment reaches a terminal or truncated state
        if terminated or truncated:
            env.reset()
    
    # Convert the collected lists into tensors
    state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    action_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
    
    return TensorDataset(state_tensor, action_tensor, next_state_tensor)
