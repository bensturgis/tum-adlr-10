import numpy as np

def compute_one_step_pred_accuracy(true_env, learned_env, num_samples):
    # List to store squared errors
    squared_errors = []

    for _ in range(num_samples):
        # Randomly sample a state within observation space bounds
        state = np.random.uniform(
            low=true_env.observation_space.low,
            high=true_env.observation_space.high
        )
        # Sample a random action from the action space
        action = true_env.action_space.sample()

        # Set the same state in both environments
        true_env.state = state
        learned_env.state = state

        # Step forward in both environments
        true_next_state, _, true_terminated, true_truncated, _ = true_env.step(action)
        pred_next_state, _, pred_terminated, pred_truncated, _ = learned_env.step(action)

        # Ensure calculations stop on termination/truncation
        if true_terminated or true_truncated or pred_terminated or pred_truncated:
            continue

        # Calculate the squared error between true and predicted next states
        squared_error = np.sum((pred_next_state - true_next_state) ** 2)
        squared_errors.append(squared_error)

    # Calculate RMSE
    if len(squared_errors) == 0:  # Handle edge case where no valid samples exist
        return None

    rmse = np.sqrt(np.mean(squared_errors))
    return rmse