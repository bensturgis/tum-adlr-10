import sys
import os
sys.path.append(os.path.abspath("."))

from utils.visualization_utils import plot_state_space_trajectory

plot_state_space_trajectory(
    experiment=9,
    sampling_method="Random Sampling Shooting",
    num_al_iterations=3,
)