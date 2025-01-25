import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.laplace_bnn import LaplaceBNN
from models.mc_dropout_bnn import MCDropoutBNN
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple

BOUND_SHRINK_FACTOR = 0.8

def plot_state_space_trajectory(
    experiment: int, sampling_method: str, num_al_iterations: int,
    true_env: gym.Env, horizon: int, repetition: int = 0,
    show_plot: bool = True
):
    """
    Plot the state space exploration for a specified experiment, sampling method,
    repetition, and number of active learning iterations.

    Args:
        experiment (int): Experiment number.
        sampling_method (str): The sampling method (Allowed options: "Random Exploration" and
                               "Random Sampling Shooting").
        num_al_iterations (int): Number of active learning iterations to plot.
        visualization_bounds (Dict[str, float]): Bounds for state space dimensions (min/max values).
        state_dims_to_vis (Tuple[int]): Indices of state dimensions to plot on x and y axes.
        state_dim_names (Dict[int, str]): Names of state dimensions for axis labels.
        state_bounds (Dict[str, float]): Dictionary specifying the bounds of the reachable state space.
        repetition (int): The repetition number. Defaults to 0.
    """
    # Construct the path to the state trajectories folder
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    experiment_path = base_path / f"experiment_{experiment}"
    trajectory_file = experiment_path / "state_trajectories" / sampling_method / f"repetition_{repetition}.npy"

    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file not found at: {trajectory_file}")

    # Load the trajectory file (shape: (num_al_iterations, horizon, state_dim))
    trajectories = np.load(trajectory_file)

    # Initialize the plot
    plt.figure(
        num=f"State Space Exploration of {sampling_method} after {num_al_iterations} Active Learning Iterations",
        figsize=(10, 10)
    )
    plt.title(f"{sampling_method} after {num_al_iterations} Active Learning Iterations")

    if true_env.name == "Mass-Spring-Damper System":
        # Get bounds for state space dimensions (min/max values)
        visualization_bounds = true_env.get_state_bounds(horizon=horizon, bound_shrink_factor=0.8)

        # Label axes
        plt.xlabel("Position")
        plt.ylabel("Velocity")

        # Plot the trajectories for each active learning iteration
        for iteration in range(num_al_iterations):
            # Check if the iteration exists in the file
            if iteration < trajectories.shape[0]:
                trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
                # plt.scatter(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}", s=10)
                plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}")

        # Set minimum and maximum values
        plt.xlim(visualization_bounds[0][0], visualization_bounds[0][1])
        plt.ylim(visualization_bounds[1][0], visualization_bounds[1][1])
    elif true_env.name == "Reacher":
        # Label axes
        plt.xlabel(r"$\theta_1$")
        plt.xlabel(r"$\theta_2$")

        # Plot the trajectories for each active learning iteration
        for iteration in range(num_al_iterations):
            # Check if the iteration exists in the file
            if iteration < trajectories.shape[0]:
                trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
                # Plot theta_1 and theta_2 instead of sine/cosine representation
                theta_1 = np.arctan2(trajectory[:, 2], trajectory[:, 0])
                theta_2 = np.arctan2(trajectory[:, 3], trajectory[:, 1])
                # plt.scatter(theta_1, theta_2, label=f"Iteration {iteration}", s=10)
                plt.plot(theta_1, theta_2, label=f"Iteration {iteration}")

        # Set minimum and maximum values
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
    else:
        raise ValueError(f"Unsupported environment: {true_env.name}. "
                         "Expected 'Mass-Spring-Damper System' or 'Reacher'.")


    plt.legend()
    plt.grid(True)
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(experiment_path / f'state_space_trajectory_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')


def plot_state_space_pred_var(
    experiment: int, sampling_method: str, num_al_iterations: int, state_bounds: Dict[str, float],
    repetition: int = 0, show_plot: bool = True, model = None
):
    """
    Plot the predictive variance over all state and action space for a Bayesian Model.
    To visualize where the model believe is "informative"
    Currently only works for state_dim=2, action_dim=1, which shows a 3D-pointcloud

    Args:
        experiment (int): Experiment number.
        sampling_method (str): The sampling method (Allowed options: "Random Exploration" and
                               "Random Sampling Shooting").
        num_al_iterations (int): Index of active learning iteration to plot.
        state_bounds (Dict[str, float]): Dictionary specifying the bounds of the reachable state space.
        repetition (int): The repetition number. Defaults to 0.
    """
    ## 1. plot uncertainty prediction
    # generate grid
    pixels_per_axis = 100
    pixels_per_action = 20
    x = np.linspace(BOUND_SHRINK_FACTOR * state_bounds["min_position"], BOUND_SHRINK_FACTOR * state_bounds["max_position"], pixels_per_axis) # x as horizontal
    v = np.linspace(BOUND_SHRINK_FACTOR * state_bounds["min_velocity"], BOUND_SHRINK_FACTOR * state_bounds["max_velocity"], pixels_per_axis) # v as vertical
    action = np.linspace(state_bounds["min_action"], state_bounds["max_action"], pixels_per_action) # average over action space
    X, V, A = np.meshgrid(x, v, action) # 3D meshgrid for calculation
    X_flat = X.ravel()
    V_flat = V.ravel()
    A_flat = A.ravel()

    # load corresponding model
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    experiment_path = base_path / f"experiment_{experiment}"
    if num_al_iterations > 0:
        model_weights_file = experiment_path / "training_results" / sampling_method / f"repetition_{repetition}" / f"iteration_{num_al_iterations}" / "model_weights.pt"
        if not model_weights_file.exists():
            raise FileNotFoundError(f"Weights file not found at: {model_weights_file}")

    if num_al_iterations == 0:
        model.reset_weights()
    else:
        model.load_state_dict(torch.load(model_weights_file))

    # load train dataset for LA BNN, and fit on the dataset.
    if isinstance(model, LaplaceBNN):
        # Load the trajectory file (shape: (num_al_iterations, horizon, state_dim + action_dim + state_dim))
        base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
        experiment_path = base_path / f"experiment_{experiment}"
        trajectory_file = experiment_path / "state_trajectories" / sampling_method / f"repetition_{repetition}.npy"
        trajectories = np.load(trajectory_file)
        # extract dataset from trajectories
        trajectories_till_now = trajectories[:num_al_iterations].reshape(-1, trajectories.shape[-1])  # Shape: (num_al_iterations * horizon, total_dim)
        states = trajectories_till_now[:, :model.state_dim]  # First `state_dim` columns
        actions = trajectories_till_now[:, model.state_dim: model.state_dim + model.action_dim]  # second `action_dim` columns
        next_states = trajectories_till_now[:, model.state_dim + model.action_dim:]  # Last `state_dim` columns
        # Convert to PyTorch tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

        # Create TensorDataset
        dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        model.fit_posterior(dataloader)

    # calculate predictive variance for each point
    states_batch = torch.tensor(np.column_stack((X_flat, V_flat)), dtype=torch.float32)  # [batch_size, state_dim]
    actions_batch = torch.tensor(A_flat[:, np.newaxis], dtype=torch.float32)  # [batch_size, action_dim]
    device = next(model.parameters()).device
    states_batch.to(device)
    actions_batch.to(device)
    _, pred_vars = model.bayesian_pred(states_batch, actions_batch)
    var_flat = np.sum(pred_vars, axis=1) # var summed over dimension [pixels_per_axis^3, 2]
    mean_var_flat = var_flat.reshape([pixels_per_axis*pixels_per_axis, pixels_per_action]).mean(axis=1) # average over action [pixels_per_axis^2,]

    # plot 2D pointcloud
    plt.figure(figsize=(12, 10))
    X, V = np.meshgrid(x, v) # 2D meshgrid for plot
    Z = mean_var_flat.reshape([pixels_per_axis, pixels_per_axis]) # map of uncertainty
    plt.pcolormesh(X, V, Z, shading='auto', cmap='viridis')
    plt.colorbar(label='uncertainty prediction')

    ## 2. plot explored trajectories till current iteration
    # Construct the path to the state trajectories folder
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    experiment_path = base_path / f"experiment_{experiment}"
    trajectory_file = experiment_path / "state_trajectories" / sampling_method / f"repetition_{repetition}.npy"
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file not found at: {trajectory_file}")
    # Load the trajectory file (shape: (num_al_iterations, horizon, state_dim))
    trajectories = np.load(trajectory_file)

    # Plot the trajectories for each active learning iteration
    for iteration in range(num_al_iterations):
        # Check if the iteration exists in the file
        if iteration < trajectories.shape[0]:
            trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
            # plt.scatter(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}", s=10)
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}")
    if num_al_iterations < trajectories.shape[0]:
        trajectory = trajectories[num_al_iterations]
        plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle="--", alpha=0.6, label=f"Iteration {num_al_iterations}")

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(f'State Space Uncertainty and explored Trajectories of Iteration {num_al_iterations}')
    plt.legend()
    plt.xlim(BOUND_SHRINK_FACTOR * state_bounds["min_position"], BOUND_SHRINK_FACTOR * state_bounds["max_position"])
    plt.ylim(BOUND_SHRINK_FACTOR * state_bounds["min_velocity"], BOUND_SHRINK_FACTOR * state_bounds["max_velocity"])
    if show_plot:
        plt.show()
    else:
        plt.savefig(experiment_path / f'state_space_pred_var_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')
