import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.mc_dropout_bnn import MCDropoutBNN
import torch

def plot_state_space_trajectory(
    experiment: int, sampling_method: str, num_al_iterations: int, repetition: int = 0,
):
    """
    Plot the state space exploration for a specified experiment, sampling method,
    repetition, and number of active learning iterations.

    Args:
        experiment (int): Experiment number.
        sampling_method (str): The sampling method (Allowed options: "Random Exploration" and
                               "Random Sampling Shooting").
        num_al_iterations (int): Number of active learning iterations to plot.
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
        figsize=(5, 5)
    )
    plt.title(f"{sampling_method} after {num_al_iterations} Active Learning Iterations")
    # TODO: Find smarter solution to label axis correctly
    plt.xlabel("x")
    plt.ylabel("v")

    # Plot the trajectories for each active learning iteration
    for iteration in range(num_al_iterations):
        # Check if the iteration exists in the file
        if iteration < trajectories.shape[0]:
            trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
            # plt.scatter(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}", s=10)
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}")

    plt.legend()
    plt.grid(True)
    # TODO: Find smarter solution to find correct ranges
    plt.xlim(-1, 1)
    plt.ylim(-3, 3)
    plt.show()


def plot_state_space_pred_var(bnn_model: MCDropoutBNN, state_space: list, action_space:list):
    """
    Plot the predictive variance over all state and action space for a Bayesian Model.
    To visualize where the model believe is "informative"
    Currently only works for state_dim=2, action_dim=1, which shows a 3D-pointcloud

    Args:
        bnn_model (MCDropoutBNN): must be a bayesian model with function bayesian_pred.
        state_space (list): manually define state range we want to visualize: [[low, high], [low, high]]
        action_space (list): manually define action range we want to visualize: [[low, high]]
    """
    # generate grid
    pixels_per_axis = 10
    x = np.linspace(state_space[0][0], state_space[0][1], pixels_per_axis) # x as horizontal
    v = np.linspace(state_space[1][0], state_space[1][1], pixels_per_axis) # v as vertical
    action = np.linspace(action_space[0][0], action_space[0][1], pixels_per_axis) # for one given action
    X, V, A = np.meshgrid(x, v, action)
    # batch_size = pixels_per_axis**3 # total number of input samples
    X_flat = X.ravel()
    V_flat = V.ravel()
    A_flat = A.ravel()

    # calculate predictive variance for each point
    states_batch = torch.tensor(np.column_stack((X_flat, V_flat)), dtype=torch.float32)  # [batch_size, state_dim]
    actions_batch = torch.tensor(A_flat[:, np.newaxis], dtype=torch.float32)  # [batch_size, action_dim]
    device = next(bnn_model.parameters()).device
    states_batch.to(device)
    actions_batch.to(device)
    _, pred_vars = bnn_model.bayesian_pred(states_batch, actions_batch)
    var_flat = np.sum(np.log(pred_vars), axis=1)

    # plot 3D pointcloud
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_flat, V_flat, A_flat, s=var_flat-np.min(var_flat), c=var_flat, cmap='viridis', alpha=0.9)
    colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label('uncertainty: ln(var)')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_zlabel('a')
    ax.set_title('3D State Space Uncertainty')
    plt.show()
