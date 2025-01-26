import sys
import os
sys.path.append(os.path.abspath("."))

from environments.mass_spring_damper_system import TrueMassSpringDamperEnv
from environments.reacher import TrueReacherEnv
from utils.visualization_utils import plot_state_space_trajectory

# Use same environment and horizon as for the active learning
# true_env = TrueMassSpringDamperEnv(noise_var=0.0)
true_env = TrueReacherEnv()
HORIZON = 100

exp_idx = 13
num_iter = 20

plot_state_space_trajectory(
    sampling_method="Random Exploration",
    experiment=exp_idx,
    repetition=0,
    num_al_iterations=num_iter,
    true_env=true_env,
    horizon=HORIZON
)