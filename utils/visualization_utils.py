import gymnasium as gym
import imageio.v2 as imageio
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
from models.laplace_bnn import LaplaceBNN
import numpy as np
from pathlib import Path
from PIL import Image
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple

from environments.mass_spring_damper_system import LearnedMassSpringDamperEnv, TrueMassSpringDamperEnv
from environments.reacher import LearnedReacherEnv, TrueReacherEnv
from metrics.evaluation_metric import EvaluationMetric
from models.laplace_bnn import LaplaceBNN
from models.mc_dropout_bnn import MCDropoutBNN
from models.feedforward_nn import FeedforwardNN

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
        plt.savefig(experiment_path / f'{sampling_method}_traj_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')


def plot_msd_uncertainty(
    experiment: int, sampling_method: str, num_al_iterations: int, true_env: gym.Env, horizon: int,
    repetition: int = 0, show_plot: bool = True, title_size:float = 10, show_colorbar: bool = True, model = None, single_action=None
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
    state_bounds = true_env.get_state_bounds(horizon=horizon, bound_shrink_factor=1.2)
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
    if show_colorbar:
        plt.figure(figsize=(6.3,5))
    else:
        plt.figure(figsize=(5,5))
    X, V = np.meshgrid(x, v) # 2D meshgrid for plot
    Z = mean_var_flat.reshape([pixels_per_axis, pixels_per_axis]) # map of uncertainty
    plt.pcolormesh(X, V, Z, shading='auto', cmap='viridis', vmax=0.2, vmin=0.02)
    if show_colorbar:
        plt.colorbar(label='uncertainty prediction')
    # plt.gca().set_aspect(1, adjustable="datalim")
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.axis("scaled")
    # plt.gca().set_adjustable("box")

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
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}", linewidth=0.9)
    if num_al_iterations < trajectories.shape[0]:
        trajectory = trajectories[num_al_iterations]
        plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle="--", alpha=0.6, label=f"Iteration {num_al_iterations}", linewidth=0.8)

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.xlim(state_bounds[0][0], state_bounds[0][1])
    plt.ylim(state_bounds[1][0], state_bounds[1][1])
    
    if title_size > 0:
        plt.title(f'{sampling_method} at Iteration {num_al_iterations}', fontsize=title_size)
    # plt.title(f'State Space Uncertainty and explored Trajectories of Iteration {num_al_iterations}')
    # plt.legend()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(experiment_path / f'{sampling_method}_var_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')


def plot_reacher_uncertainty(
    experiment: int, sampling_method: str, num_al_iterations: int, true_env: gym.Env, horizon: int,
    repetition: int = 0, show_plot: bool = True, show_colorbar: bool = True, title_size:float = 10, model = None
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
    state_bounds = true_env.get_state_bounds(horizon=horizon, bound_shrink_factor=0.06)
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
    plt.figure()
    plt.pcolormesh(T1, T2, Z, shading='auto', cmap='viridis', vmax=1.2, vmin=0.2)
    if show_colorbar:
        plt.colorbar(label='uncertainty prediction')
    plt.axis("equal")
    plt.gca().set_adjustable("box")

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
            plt.scatter(theta_1, theta_2, label=f"Iteration {iteration}", s=3)
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

    if title_size > 0:
        plt.title(f'{sampling_method} at Iteration {num_al_iterations}', fontsize=title_size)
    # plt.legend()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(experiment_path / f'{sampling_method}_var_rep{repetition}_iter{num_al_iterations}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_state_distribution(
    dataset: TensorDataset, 
    dim_idx: int, 
    state_dim_names: Dict[int, str]
) -> None:
    """
    Plots a histogram of the values in the dataset for the specified dimension index.
    
    Args:
        dataset (TensorDataset): A dataset containing states.
        dim_idx (int): Index of the state dimension to plot.
        state_dim_names (Dict[int, str]): A dictionary mapping dimension indices
            to human-readable dimension names.
    """    
    # Extract the first tensor in the dataset, which should contain states of shape (N, state_dim)
    states = dataset.tensors[0]  # shape: [N x state_dim]

    # Convert the requested dimension to a NumPy array
    dim_data = states[:, dim_idx].numpy()
    
    # Create a label from the dim index or use a fallback if not found
    dim_label = state_dim_names.get(dim_idx, f"Dimension {dim_idx}")
    
    # Plot the histogram
    plt.figure(figsize=(6,4))
    low_percentile = np.percentile(dim_data, 2.5)
    high_percentile = np.percentile(dim_data, 97.5)
    plt.hist(dim_data, bins=50, range=(low_percentile, high_percentile))
    plt.title(f"Distribution of {dim_label}")
    plt.xlabel(dim_label)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

from PIL import Image, ImageSequence
from PIL import Image, ImageSequence
import re
import imageio.v2 as imageio
def generate_GIF(experiment: int, sampling_method: str, repetition: int = 0):
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    image_path = base_path / f"experiment_{experiment}"

    # read imgs
    image_files = list(image_path.glob(f"{sampling_method}_var_rep{repetition}_iter*.png"))
    def extract_iteration(filepath):
        filename = str(filepath)
        match = re.search(r"iter(\d+)", filename)
        return int(match.group(1)) if match else float('inf')
    image_files = sorted(image_files, key=extract_iteration) # sort in iterations
    images = [Image.open(img) for img in image_files] # open images
    for i in range(5):
        images.append(images[-1])

    # save gif
    output_path = image_path / f"{sampling_method}_var_rep{repetition}.gif"
    imageio.mimsave(output_path, [img.convert('RGB') for img in images], duration=300, format='GIF', loop=0)

    print(f"GIF Saved: {output_path}")

def get_horizon(experiment: int) -> int:
    # Construct the path to the state experiment folder
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    experiment_path = base_path / f"experiment_{experiment}"
    
    # Extract hyperparameters
    hyperparams_file = experiment_path / "hyperparameters.json"
    with open(hyperparams_file, 'r') as file:
        hyperparams = json.load(file)

        # TODO: assert that horizons of all sampling methods match
        horizon = hyperparams["sampling_methods"][0]["horizon"]

    return horizon

def reconstruct_envs(
    experiment: int, device: torch.device = torch.device('cpu')
) -> Tuple[gym.Env, gym.Env]:
    # Construct the path to the state experiment folder
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    experiment_path = base_path / f"experiment_{experiment}"
    
    # Extract hyperparameters
    hyperparams_file = experiment_path / "hyperparameters.json"
    with open(hyperparams_file, 'r') as file:
        hyperparams = json.load(file)

        # Extract hyperparameters of true environment
        if hyperparams["true_env"]["name"] == "Mass-Spring-Damper System":
            mass = hyperparams["true_env"]["mass"]
            stiffness = hyperparams["true_env"]["stiffness"]
            damping = hyperparams["true_env"]["damping"]
            time_step = hyperparams["true_env"]["time_step"]
            non_linear = hyperparams["true_env"]["nonlinear"]
            noise_var = hyperparams["true_env"]["noise_var"]
            true_env = TrueMassSpringDamperEnv(
                mass=mass,
                stiffness=stiffness,
                damping=damping,
                time_step=time_step,
                nonlinear=non_linear,
                noise_var=noise_var
            )
        elif hyperparams["true_env"]["name"] == "Reacher":
            link_length = hyperparams["true_env"]["link_length"]
            time_step = hyperparams["true_env"]["time_step"]
            noise_var = hyperparams["true_env"].get("noise_var", 0.0)
            true_env = TrueReacherEnv(
                link_length=link_length,
                time_step=time_step,
                noise_var=noise_var
            )
        else:
            raise ValueError(f"Unknown environment: {hyperparams['true_env']['name']}. "
                             "Supported environments are 'Mass-Spring-Damper System' and 'Reacher'.")
        print(f"True Environment: {true_env.params_to_dict()}")

        # Extract hyperparameters of learned environment
        if hyperparams["learned_env"]["model"]["name"] == "Monte Carlo Dropout Bayesian Neural Network":
            state_bounds = hyperparams["learned_env"]["model"]["state_bounds"]
            state_bounds = {
                int(key): list(map(float, value.strip("[]").split())) for key, value in state_bounds.items()
            }
            state_dim = hyperparams["learned_env"]["model"].get("state_dim", len(state_bounds))
            action_bounds = hyperparams["learned_env"]["model"]["action_bounds"]
            action_bounds = {
                int(key): list(map(float, value.strip("[]").split())) for key, value in action_bounds.items()
            }
            action_dim = hyperparams["learned_env"]["model"].get("action_dim", len(action_bounds))
            input_expansion = hyperparams["learned_env"]["model"]["input_expansion"]
            hidden_size = hyperparams["learned_env"]["model"].get("hidden_size", 72)
            drop_prob = hyperparams["learned_env"]["model"].get("drop_prob", 0.1)

            dynamics_model = MCDropoutBNN(
                state_dim=state_dim,
                action_dim=action_dim,
                input_expansion=input_expansion,
                state_bounds=state_bounds,
                action_bounds=action_bounds,
                hidden_size=hidden_size,
                drop_prob=drop_prob,
                device=device,
            )
        elif hyperparams["learned_env"]["model"]["name"] == "Laplace Approximation Bayesian Neural Network":
            state_bounds = hyperparams["learned_env"]["model"]["state_bounds"]
            state_bounds = {
                int(key): list(map(float, value.strip("[]").split())) for key, value in state_bounds.items()
            }
            state_dim = hyperparams["learned_env"]["model"].get("state_dim", len(state_bounds))
            action_bounds = hyperparams["learned_env"]["model"]["action_bounds"]
            action_bounds = {
                int(key): list(map(float, value.strip("[]").split())) for key, value in action_bounds.items()
            }
            action_dim = hyperparams["learned_env"]["model"].get("action_dim", len(action_bounds))
            input_expansion = hyperparams["learned_env"]["model"]["input_expansion"]
            hidden_size = hyperparams["learned_env"]["model"].get("hidden_size", 72)
            print(f"State bounds: {state_bounds}")
            dynamics_model = LaplaceBNN(
                state_dim=state_dim,
                action_dim=action_dim,
                input_expansion=true_env.input_expansion,
                state_bounds=state_bounds,
                action_bounds=action_bounds,
                hidden_size=hidden_size,
                device=device,
            )
        elif hyperparams["learned_env"]["model"]["name"] == "Standard Feedforward Neural Network":
            state_dim = hyperparams["learned_env"]["model"]["state_dim"]
            action_dim = hyperparams["learned_env"]["model"]["action_dim"]
            hidden_size = hyperparams["learned_env"]["model"]["hidden_size"]

            dynamics_model = FeedforwardNN(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_size=hidden_size
            )
        else:
            raise ValueError(f"Unknown model: {hyperparams['learned_env']['model']['name']}. "
                             "Supported models are 'Monte Carlo Dropout Bayesian Neural Network', "
                             "'Laplace Approximation Bayesian Neural Network' and 'Standard "
                             "Feedforward Neural Network'.")

        if hyperparams["learned_env"]["name"] == "Mass-Spring-Damper System":
            learned_env = LearnedMassSpringDamperEnv(
                model=dynamics_model
            )
        elif hyperparams["learned_env"]["name"] == "Reacher":
            learned_env = LearnedReacherEnv(
                model=dynamics_model
            )
        else:
            raise ValueError(f"Unknown environment: {hyperparams['learned_env']['name']}. "
                             "Supported environments are 'Mass-Spring-Damper System' and 'Reacher'.")
        print(f"Learned Environment: {learned_env.params_to_dict()}")

        return true_env, learned_env

def plot_prediction_error(
        experiment: int, learned_env: gym.Env, metrics: List[EvaluationMetric],
        num_al_iterations: int = None, eval_repetitions: List[int] = None,
        plot_variances: bool = True, show: bool = True, save: bool = True,
        device: torch.device = torch.device('cpu')
    ):
    # Construct the path to the state experiment folder
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    experiment_path = base_path / f"experiment_{experiment}"
    
    # Extract hyperparameters
    hyperparams_file = experiment_path / "hyperparameters.json"
    with open(hyperparams_file, 'r') as file:
        hyperparams = json.load(file)

        # Extract used sampling methods
        sampling_methods = [method["name"] for method in hyperparams["sampling_methods"]]
        if num_al_iterations is None:
            num_al_iterations = hyperparams["num_al_iterations"]
        if eval_repetitions is None:
            num_eval_repetitions = hyperparams["num_eval_repetitions"]
            eval_repetitions = range(num_eval_repetitions)
        else:
            num_eval_repetitions = len(eval_repetitions)

    mean_errors = {}
    std_errors = {}
    for sampling_method in sampling_methods:
        mean_errors[sampling_method] = {}
        std_errors[sampling_method] = {}
        for metric in metrics:
            errors = np.zeros((num_eval_repetitions, num_al_iterations))
            for rep_idx, rep in enumerate(eval_repetitions):
                for iter in range(num_al_iterations):
                    training_path = (
                        experiment_path / "training_results" / sampling_method 
                        / f"repetition_{rep}" / f"iteration_{iter+1}"
                    )

                    model_weights_path = training_path / "model_weights.pt"
                    
                    learned_env.model.load_state_dict(torch.load(
                        model_weights_path, map_location=device
                    ))
                    learned_env.model.eval()

                    errors[rep_idx][iter] = metric.evaluate()

            mean_errors[sampling_method][metric.name] = list(np.mean(errors, axis=0))
            std_errors[sampling_method][metric.name] = list(np.std(errors, axis=0))

    figures = create_prediction_error_plot(
        mean_errors=mean_errors,
        std_errors=std_errors,
        num_al_iterations=num_al_iterations,
        env_name=learned_env.name,
        plot_variances=plot_variances
    )

    # Save the error plots
    if save:
        for metric in metrics:
            plot_path = experiment_path / f"{metric.name.lower().replace(' ', '_').replace('-', '_')}_plot.png"
            counter = 2

            # Check if the file exists and append _01, _02, etc.
            while plot_path.exists():
                plot_path = experiment_path / f"{metric.name.lower().replace(' ', '_').replace('-', '_')}_plot_{counter:02d}.png"
                counter += 1

            figures[metric.name].savefig(plot_path, bbox_inches="tight", dpi=300)
            print(f"{metric.name} plot saved to {plot_path}.")

    if show:
        plt.show()

def create_prediction_error_plot(
    mean_errors: Dict[str, Dict[str, List[float]]],
    std_errors: Dict[str, Dict[str, List[float]]],
    num_al_iterations: int, env_name: str, 
    plot_variances: bool = True
) -> Dict[str, plt.Figure]:
    """
    Plots the mean and standard deviation of prediction errors for different sampling
    methods over active learning iterations.

    Args:
        mean_errors (Dict[str, Dict[str, List]]): A nested dictionary where
            mean_errors[sampling_method][metric] is a list of length
            `num_al_iterations` with mean errors per iteration across repetitions.
        std_errors (Dict[str, Dict[str, List]]): A nested dictionary where
            std_errors[sampling_method][metric] is a list of length 
            `num_al_iterations` with standard deviations per iteration across repetitions.
    """
    # Define map of colors to distinguish the sampling methods
    color_map = {
        "Random Exploration": "blue",
        "Random Sampling Shooting": "green",
        "Soft Actor Critic": "orange"
    }
    
    iterations = range(1, num_al_iterations + 1)
    sampling_methods = list(mean_errors.keys())
    evaluation_metrics = list(next(iter(mean_errors.values())).keys())

    # Create a plot for each evaluation metric
    figures = {}
    for metric in evaluation_metrics:
        fig, ax = plt.subplots(
            num=f"{metric} over Active Learning Iterations", figsize=(8, 5)
        )
        
        # Plot mean and std for each sampling method using the specified color
        for sampling_method in sampling_methods:
            # Extract arrays without overwriting the original dictionary variables
            mean_arr = mean_errors[sampling_method][metric]
            std_arr = std_errors[sampling_method][metric]

            # Lookup the color for this method from the map
            color = color_map.get(sampling_method, "black")  # default to black if not found

            # Plot the mean curve
            ax.plot(iterations, mean_arr, label=sampling_method, color=color)

            # Plot individual data points for every iteration
            ax.scatter(iterations, mean_arr, color=color, alpha=1.0, marker='o')

            # Plot error bars (±1 std) if plot_variances is True
            if plot_variances:
                ax.errorbar(
                    iterations, mean_arr,
                    yerr=std_arr, fmt='o',
                    color=color, ecolor=color,
                    capsize=3, elinewidth=1
                )

            # Set integer x-axis labels
            ax.xaxis.set_major_locator(MultipleLocator(5))

            # Increase the size of the numbers on x- and y-axis
            ax.tick_params(axis='both', which='major', labelsize=12)

        # Customize axes and title
        ax.set_xlabel("Active Learning Iteration", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f"{env_name}: {metric}", fontsize=16, pad=14)
        ax.legend(fontsize=14)
        ax.grid(True)

        figures[metric] = fig

    return figures

def compare_GIF(experiment: int, repetition: int = 0):
    # load corresponding model
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    image_path = base_path / f"experiment_{experiment}"

    gif1 = Image.open(image_path / f"Random Exploration_var_rep{repetition}.gif")
    gif2 = Image.open(image_path / f"Random Sampling Shooting_var_rep{repetition}.gif")
    gif3 = Image.open(image_path / f"Soft Actor Critic_var_rep{repetition}.gif")

    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]
    frames3 = [frame.copy() for frame in ImageSequence.Iterator(gif3)]

    min_frames = min(len(frames1), len(frames2), len(frames3))
    frames1, frames2, frames3 = frames1[:min_frames], frames2[:min_frames], frames3[:min_frames]

    new_frames = []
    for f1, f2, f3 in zip(frames1, frames2, frames3):
        width, height = f1.size
        width_last, height = f3.size
        new_frame = Image.new("RGB", (width * 2 + width_last, height))
        new_frame.paste(f1, (0, 0))
        new_frame.paste(f2, (width, 0))
        new_frame.paste(f3, (width * 2, 0))
        new_frames.append(new_frame)
    for i in range(8):
        new_frames.append(new_frames[-1])

    output_path = image_path / f"comparison_rep{repetition}.gif"
    imageio.mimsave(output_path, [img.convert('RGB') for img in new_frames], duration=300, format='GIF', loop=0)

    print(f"combined GIF saved to: {output_path}")


def compare_png(experiment: int, repetition: int = 0, iteration: int = 25):
    # load corresponding model
    base_path = Path(__file__).parent.parent / "experiments" / "active_learning_evaluations"
    image_path = base_path / f"experiment_{experiment}"

    png1 = Image.open(image_path / f"Random Exploration_var_rep{repetition}_iter{iteration}.png")
    png2 = Image.open(image_path / f"Random Sampling Shooting_var_rep{repetition}_iter{iteration}.png")
    png3 = Image.open(image_path / f"Soft Actor Critic_var_rep{repetition}_iter{iteration}.png")

    width, height = png1.size
    width_last, height = png3.size
    mosaic = Image.new("RGB", (width * 2 + width_last, height))
    mosaic.paste(png1, (0, 0))
    mosaic.paste(png2, (width, 0))
    mosaic.paste(png3, (width * 2, 0))

    output_path = image_path / f"comparison_rep{repetition}_iter{iteration}.png"
    mosaic.save(output_path, format="PNG")

    print(f"combined var plot saved to: {output_path}")