import sys
import os
sys.path.append(os.path.abspath("."))

from dynamical_systems.mass_spring_damper_system import TrueMassSpringDamperEnv, LearnedMassSpringDamperEnv
from utils.visualization_utils import plot_state_space_pred_var
from models.laplace_bnn import LaplaceBNN
from models.mc_dropout_bnn import MCDropoutBNN

# Use same environment and horizon as for the active learning
env = TrueMassSpringDamperEnv(noise_var=0.0)
HORIZON = 50

state_bounds = env.compute_state_bounds(horizon=HORIZON)
# Hyperparameters for neural network
STATE_DIM = 2
ACTION_DIM = 1
HIDDEN_SIZE = 72          # Hidden units in the neural network
DROP_PROB = 0.1           # Dropout probability for the bayesian neural network
DEVICE = "cuda"
# Initialize trained dynamics model and load saved weights
# bnn_model = MCDropoutBNN(STATE_DIM, ACTION_DIM, hidden_size=HIDDEN_SIZE, drop_prob=DROP_PROB, device=DEVICE)
bnn_model = LaplaceBNN(STATE_DIM, ACTION_DIM, hidden_size=HIDDEN_SIZE, device=DEVICE)
exp_idx = 13
num_iter = 20

for num_iter in range(4,22,4):
    plot_state_space_pred_var(
        sampling_method="Random Sampling Shooting",
        experiment=exp_idx,
        repetition=4,
        num_al_iterations=num_iter,
        state_bounds=state_bounds,
        model=bnn_model,
        show_plot=False
    )