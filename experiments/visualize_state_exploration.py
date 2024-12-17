import sys
import os
sys.path.append(os.path.abspath("."))

from dynamical_systems.mass_spring_damper_system import MassSpringDamperEnv
from utils.visualization_utils import plot_state_space_trajectory

# Use same environment and horizon as for the active learning
env = MassSpringDamperEnv(noise_var=0.0)
HORIZON = 50

state_bounds = env.compute_state_bounds(horizon=HORIZON)
exp_idx = 2

plot_state_space_trajectory(
    sampling_method="Random Sampling Shooting",
    experiment=exp_idx,
    repetition=0,
    num_al_iterations=15,
    state_bounds=state_bounds
)

plot_state_space_trajectory(
    sampling_method="Random Exploration",
    experiment=exp_idx,
    repetition=0,
    num_al_iterations=15,
    state_bounds=state_bounds
)