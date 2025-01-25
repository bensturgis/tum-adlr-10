import sys
import os
sys.path.append(os.path.abspath("."))

from environments.mass_spring_damper_system import TrueMassSpringDamperEnv, LearnedMassSpringDamperEnv
from utils.visualization_utils import plot_state_space_trajectory

# Use same environment and horizon as for the active learning
env = TrueMassSpringDamperEnv(noise_var=0.0)
HORIZON = 50

state_bounds = env.compute_state_bounds(horizon=HORIZON)
exp_idx = 17
num_iter = 20

# for num_iter in range(10):
plot_state_space_trajectory(
    sampling_method="Soft Actor Critic",
    experiment=exp_idx,
    repetition=0,
    num_al_iterations=num_iter,
    state_bounds=state_bounds
)

plot_state_space_trajectory(
    sampling_method="Random Exploration",
    experiment=exp_idx,
    repetition=0,
    num_al_iterations=num_iter,
    state_bounds=state_bounds
)