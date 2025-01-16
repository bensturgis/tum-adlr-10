import sys
import os
sys.path.append(os.path.abspath("."))

from dynamical_systems.mass_spring_damper_system import TrueMassSpringDamperEnv, LearnedMassSpringDamperEnv
from utils.visualization_utils import plot_state_space_pred_var

# Use same environment and horizon as for the active learning
env = TrueMassSpringDamperEnv(noise_var=0.0)
HORIZON = 50

state_bounds = env.compute_state_bounds(horizon=HORIZON)
exp_idx = 8

plot_state_space_pred_var(
    sampling_method="Random Sampling Shooting",
    experiment=exp_idx,
    repetition=0,
    num_al_iterations=5,
    state_bounds=state_bounds,
    show_plot=True
)