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
        repetition (int): The repetition number. Defaults to 0.
        true_env (gym.Env): The environment instance, which determines how trajectories are plotted.
        horizon (int): The number of steps used for computing state bounds.
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
        # # Label axes
        # plt.xlabel(r"$\theta_1$")
        # plt.ylabel(r"$\theta_2$")

        # # Plot the trajectories for each active learning iteration
        # for iteration in range(num_al_iterations):
        #     # Check if the iteration exists in the file
        #     if iteration < trajectories.shape[0]:
        #         trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
        #         # Plot theta_1 and theta_2 instead of sine/cosine representation
        #         theta_1 = np.arctan2(trajectory[:, 2], trajectory[:, 0])
        #         theta_2 = np.arctan2(trajectory[:, 3], trajectory[:, 1])
        #         plt.scatter(theta_1, theta_2, label=f"Iteration {iteration}", s=10)
        #         # plt.plot(theta_1, theta_2, label=f"Iteration {iteration}")

        # Label axes
        plt.xlabel(r"x")
        plt.ylabel(r"y")

        # Plot the trajectories for each active learning iteration
        for iteration in range(num_al_iterations):
            # Check if the iteration exists in the file
            if iteration < trajectories.shape[0]:
                trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
                # Plot theta_1 and theta_2 instead of sine/cosine representation
                theta_1 = np.arctan2(trajectory[:, 2], trajectory[:, 0])
                theta_2 = np.arctan2(trajectory[:, 3], trajectory[:, 1])
                l = true_env.link_length
                x = l*(np.cos(theta_1) - np.cos(theta_1+theta_2))
                y = l*(np.sin(theta_1) + np.sin(theta_1+theta_2))
                # plt.scatter(x, y, label=f"Iteration {iteration}", s=10)
                plt.plot(x, y, label=f"Iteration {iteration}")

        # # Set minimum and maximum values
        # plt.xlim(-np.pi, np.pi)
        # plt.ylim(-np.pi, np.pi)
    else:
        raise ValueError(f"Unsupported environment: {true_env.name}. "
                         "Expected 'Mass-Spring-Damper System' or 'Reacher'.")

    plt.legend()
    plt.grid(True)
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(experiment_path / f'state_space_trajectory_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')


def plot_msd_uncertainty(
    experiment: int, sampling_method: str, num_al_iterations: int, true_env: gym.Env, horizon: int,
    repetition: int = 0, show_plot: bool = True, model = None, single_action=None
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
    state_bounds = true_env.get_state_bounds(horizon=horizon, bound_shrink_factor=0.8)
    action_bounds = true_env.get_action_bounds()
    # generate grid
    pixels_per_axis = 100
    pixels_per_action = 20
    x = np.linspace(state_bounds[0][0], state_bounds[0][1], pixels_per_axis) # x as horizontal
    v = np.linspace(state_bounds[1][0], state_bounds[1][1], pixels_per_axis) # v as vertical
    action = np.linspace(action_bounds[0][0], action_bounds[0][1], pixels_per_action) # average over action space
    if single_action is not None:
        pixels_per_action = 1
        action = np.array([single_action])
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

    # plot 2D heatmap
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
    plt.xlim(state_bounds[0][0], state_bounds[0][1])
    plt.ylim(state_bounds[1][0], state_bounds[1][1])
    
    plt.title(f'State Space Uncertainty and explored Trajectories of Iteration {num_al_iterations}')
    plt.legend()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(experiment_path / f'state_space_pred_var_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')


def plot_reacher_uncertainty(
    experiment: int, sampling_method: str, num_al_iterations: int, true_env: gym.Env, horizon: int,
    repetition: int = 0, show_plot: bool = True, model = None
):
    """
    Plot the predictive variance over all state and action space for a Bayesian Model.
    To visualize where the model believe is "informative"
    Currently only works for state_dim=4, action_dim=2, average over axis other than theta1,theta2 / x,y

    Args:
        experiment (int): Experiment number.
        sampling_method (str): The sampling method (Allowed options: "Random Exploration" and
                               "Random Sampling Shooting").
        num_al_iterations (int): Index of active learning iteration to plot.
        state_bounds (Dict[str, float]): Dictionary specifying the bounds of the reachable state space.
        repetition (int): The repetition number. Defaults to 0.
    """
    ## 1. plot uncertainty prediction
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

    # generate grid
    state_bounds = true_env.get_state_bounds(horizon=horizon, bound_shrink_factor=0.8)
    action_bounds = true_env.get_action_bounds()
    pixels_per_axis = 100
    pixels_per_other_axes = 5
    theta_1 = np.linspace(-np.pi, np.pi, pixels_per_axis) # theta1 as horizontal
    theta_2 = np.linspace(-np.pi, np.pi, pixels_per_axis) # theta2 as vertical
    T1, T2 = np.meshgrid(theta_1, theta_2) # 3D meshgrid for calculation
    T1_flat = T1.ravel()
    T2_flat = T2.ravel()

    # calculate predictive variance for each point
    velocity_x = np.linspace(state_bounds[4][0], state_bounds[4][1], pixels_per_other_axes)
    velocity_y = np.linspace(state_bounds[5][0], state_bounds[5][1], pixels_per_other_axes)
    action_1 = np.linspace(action_bounds[0][0], action_bounds[0][1], pixels_per_other_axes)
    action_2 = np.linspace(action_bounds[1][0], action_bounds[0][1], pixels_per_other_axes)
    redundant_axes = [velocity_x, velocity_y, action_1, action_2] # list for recusion
    sum_of_redundant_samples = np.prod([len(axis) for axis in redundant_axes])
    Z = np.zeros([pixels_per_axis, pixels_per_axis]) # initialize an T1*T2 shaped array
    # use resursive function to loop over all redundant axes
    def recursion_axis(r_list, item_list=[]):
        nonlocal Z
        if len(item_list) == len(r_list):
            # initialize an T1*T2 shaped array outside
            # for each redundant variable combination, construct states/actions_batch
            states_batch = torch.tensor(np.column_stack((np.cos(T1_flat), np.cos(T2_flat), np.sin(T1_flat), np.sin(T2_flat), 
                                                         np.repeat(item_list[0],pixels_per_axis*pixels_per_axis), 
                                                         np.repeat(item_list[1],pixels_per_axis*pixels_per_axis))), dtype=torch.float32)  # [batch_size, state_dim]
            actions_batch = torch.tensor(np.column_stack((np.repeat(item_list[2],pixels_per_axis*pixels_per_axis), 
                                                          np.repeat(item_list[3],pixels_per_axis*pixels_per_axis))), dtype=torch.float32)  # [batch_size, action_dim]
            device = next(model.parameters()).device
            states_batch.to(device)
            actions_batch.to(device)
            _, pred_vars = model.bayesian_pred(states_batch, actions_batch)
            var_flat = np.sum(pred_vars, axis=1) # var summed over dimension [pixels_per_axis^3, 2]
            mean_var_flat = var_flat.reshape([pixels_per_axis*pixels_per_axis])
            Z = Z + mean_var_flat.reshape([pixels_per_axis, pixels_per_axis])/sum_of_redundant_samples # map of uncertainty

        else:
            for item in r_list[len(item_list)]:
                recursion_axis(r_list, item_list+[item])
    # run recursion
    recursion_axis(redundant_axes)

    # plot 2D heatmap
    plt.figure(figsize=(12, 10))
    plt.pcolormesh(T1, T2, Z, shading='auto', cmap='viridis')
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
    # Label axes
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")

    # Plot the trajectories for each active learning iteration
    for iteration in range(num_al_iterations):
        # Check if the iteration exists in the file
        if iteration < trajectories.shape[0]:
            trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
            # Plot theta_1 and theta_2 instead of sine/cosine representation
            theta_1 = np.arctan2(trajectory[:, 2], trajectory[:, 0])
            theta_2 = np.arctan2(trajectory[:, 3], trajectory[:, 1])
            plt.scatter(theta_1, theta_2, label=f"Iteration {iteration}", s=10)
            # plt.plot(theta_1, theta_2, label=f"Iteration {iteration}")
    # # Set minimum and maximum values
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    # Label axes
    # plt.xlabel(r"x")
    # plt.ylabel(r"y")

    # Plot the trajectories for each active learning iteration
    # for iteration in range(num_al_iterations):
    #     # Check if the iteration exists in the file
    #     if iteration < trajectories.shape[0]:
    #         trajectory = trajectories[iteration]  # Shape: (horizon, state_dim)
    #         # Plot theta_1 and theta_2 instead of sine/cosine representation
    #         theta_1 = np.arctan2(trajectory[:, 2], trajectory[:, 0])
    #         theta_2 = np.arctan2(trajectory[:, 3], trajectory[:, 1])
    #         l = true_env.link_length
    #         x = l*(np.cos(theta_1) - np.cos(theta_1+theta_2))
    #         y = l*(np.sin(theta_1) + np.sin(theta_1+theta_2))
    #         # plt.scatter(x, y, label=f"Iteration {iteration}", s=10)
    #         plt.plot(x, y, label=f"Iteration {iteration}")

    
    plt.title(f'State Space Uncertainty and explored Trajectories of Iteration {num_al_iterations}')
    plt.legend()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(experiment_path / f'state_space_pred_var_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')
