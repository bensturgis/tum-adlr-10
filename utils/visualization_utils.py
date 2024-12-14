import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
            plt.scatter(trajectory[:, 0], trajectory[:, 1], label=f"Iteration {iteration}", s=10)

    plt.legend()
    plt.grid(True)
    # TODO: Find smarter solution to find correct ranges
    plt.xlim(-1, 1)
    plt.ylim(-3, 3)
    plt.show()