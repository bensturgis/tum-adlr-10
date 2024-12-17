import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.mc_dropout_bnn import MCDropoutBNN
import torch
from typing import Dict

BOUND_SHRINK_FACTOR = 0.8

def plot_state_space_trajectory(
    experiment: int, sampling_method: str, num_al_iterations: int, state_bounds: Dict[str, float],
    repetition: int = 0,
):
    """
    Plot the state space exploration for a specified experiment, sampling method,
    repetition, and number of active learning iterations.

    Args:
        experiment (int): Experiment number.
        sampling_method (str): The sampling method (Allowed options: "Random Exploration" and
                               "Random Sampling Shooting").
        num_al_iterations (int): Number of active learning iterations to plot.
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
    # TODO: Find smarter solution to label axis correctly
    plt.xlabel("Position")
    plt.ylabel("Velocity")

    # Plot the trajectories for each active learning iteration
    for iteration in range(num_al_iterations):
        # Check if the iteration exists in the file
        if iteration < trajectories.shape[0]:
            trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
            # plt.scatter(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}", s=10)
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}")

    plt.legend()
    plt.grid(True)
    plt.xlim(BOUND_SHRINK_FACTOR * state_bounds["min_position"], BOUND_SHRINK_FACTOR * state_bounds["max_position"])
    plt.ylim(BOUND_SHRINK_FACTOR * state_bounds["min_velocity"], BOUND_SHRINK_FACTOR * state_bounds["max_velocity"])
    plt.show()


def plot_state_space_pred_var(
    experiment: int, sampling_method: str, num_al_iterations: int, state_bounds: Dict[str, float],
    repetition: int = 0, show_plot=True
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
    pixels_per_axis = 40
    pixels_per_action = 10
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
    model_weights_file = experiment_path / "training_results" / sampling_method / f"repetition_{repetition}" / f"iteration_{num_al_iterations}" / "model_weights.pt"
    if not model_weights_file.exists():
        raise FileNotFoundError(f"Weights file not found at: {model_weights_file}")
    # Hyperparameters for neural network
    STATE_DIM = 2
    ACTION_DIM = 1
    HIDDEN_SIZE = 72          # Hidden units in the neural network
    DROP_PROB = 0.1           # Dropout probability for the bayesian neural network
    # Initialize trained dynamics model and load saved weights
    bnn_model = MCDropoutBNN(STATE_DIM, ACTION_DIM, hidden_size=HIDDEN_SIZE, drop_prob=DROP_PROB)
    bnn_model.reset_weights()
    # bnn_model.load_state_dict(torch.load(model_weights_file))

    # calculate predictive variance for each point
    states_batch = torch.tensor(np.column_stack((X_flat, V_flat)), dtype=torch.float32)  # [batch_size, state_dim]
    actions_batch = torch.tensor(A_flat[:, np.newaxis], dtype=torch.float32)  # [batch_size, action_dim]
    device = next(bnn_model.parameters()).device
    states_batch.to(device)
    actions_batch.to(device)
    _, pred_vars = bnn_model.bayesian_pred(states_batch, actions_batch)
    var_flat = np.sum(np.log(pred_vars), axis=1) # var summed over dimension [pixels_per_axis^3, 2]
    mean_var_flat = var_flat.reshape([pixels_per_axis*pixels_per_axis, pixels_per_action]).mean(axis=1) # average over action [pixels_per_axis^2,]

    # plot 2D pointcloud
    plt.figure(figsize=(12, 10))
    X, V = np.meshgrid(x, v) # 2D meshgrid for plot
    X_flat = X.ravel()
    V_flat = V.ravel()
    plt.scatter(X_flat, V_flat, s=6*(mean_var_flat-np.min(mean_var_flat)), c=mean_var_flat, cmap='viridis', alpha=0.9)
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

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(f'State Space Uncertainty and explored Trajectories of Iteration {num_al_iterations}')
    plt.legend()
    plt.xlim(BOUND_SHRINK_FACTOR * state_bounds["min_position"], BOUND_SHRINK_FACTOR * state_bounds["max_position"])
    plt.ylim(BOUND_SHRINK_FACTOR * state_bounds["min_velocity"], BOUND_SHRINK_FACTOR * state_bounds["max_velocity"])
    plt.show()
