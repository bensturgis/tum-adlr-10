import sys
import os
sys.path.append(os.path.abspath("."))

from environments.mass_spring_damper_system import TrueMassSpringDamperEnv
from environments.reacher import TrueReacherEnv
from utils.visualization_utils import *
from models.laplace_bnn import LaplaceBNN
from models.mc_dropout_bnn import MCDropoutBNN
import numpy as np

'''
for MSD
'''
# # Use same environment and horizon as for the active learning
# env = TrueMassSpringDamperEnv(noise_var=0.0)
# HORIZON = 100

# state_bounds = env.get_state_bounds(horizon=HORIZON)
# actions_bounds = env.get_action_bounds()
# # Hyperparameters for neural network
# STATE_DIM = 2
# ACTION_DIM = 1
# HIDDEN_SIZE = 72          # Hidden units in the neural network
# DROP_PROB = 0.1           # Dropout probability for the bayesian neural network
# DEVICE = "cuda"
# # Initialize trained dynamics model and load saved weights
# dynamics_model = LaplaceBNN(
#     state_dim=STATE_DIM,
#     action_dim=ACTION_DIM,
#     input_expansion=env.input_expansion,
#     state_bounds=state_bounds,
#     action_bounds=actions_bounds,
#     hidden_size=HIDDEN_SIZE,
#     device=DEVICE,
# )
# exp_idx = 40
# rep = 1
# sampling_methods = ["Random Exploration", "Random Sampling Shooting", "Soft Actor Critic"]
# for sp_mtd in sampling_methods:
#     for num_iter in range(1,26,1):
#         plot_msd_uncertainty(
#             experiment=exp_idx,
#             sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
#             num_al_iterations=num_iter,
#             true_env=env,
#             horizon=HORIZON,
#             repetition=rep,
#             show_plot=False,
#             model=dynamics_model
#         )
#     generate_GIF(
#         experiment=exp_idx, 
#         sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
#         repetition=rep
#         )


'''
for Reacher
'''
# # Use same environment and horizon as for the active learning
env = TrueReacherEnv(noise_var=0.0)
HORIZON = 50

state_bounds = env.get_state_bounds(horizon=HORIZON, bound_shrink_factor=0.06)
actions_bounds = env.get_action_bounds()
# Hyperparameters for neural network
STATE_DIM = 6
ACTION_DIM = 2
HIDDEN_SIZE = 72          # Hidden units in the neural network
DROP_PROB = 0.1           # Dropout probability for the bayesian neural network
DEVICE = "cuda"
# Initialize trained dynamics model and load saved weights
# bnn_model = MCDropoutBNN(STATE_DIM, ACTION_DIM, hidden_size=HIDDEN_SIZE, drop_prob=DROP_PROB, device=DEVICE)
dynamics_model = LaplaceBNN(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    input_expansion=env.input_expansion,
    state_bounds=state_bounds,
    action_bounds=actions_bounds,
    hidden_size=HIDDEN_SIZE,
    device=DEVICE,
)

exp_idx = 41
rep = 0
sampling_methods = ["Random Exploration", "Random Sampling Shooting", "Soft Actor Critic"]
for sp_mtd in sampling_methods:
    for num_iter in range(1,26,1):
        plot_reacher_uncertainty(
            experiment=exp_idx,
            sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
            num_al_iterations=num_iter,
            true_env=env,
            horizon=HORIZON,
            repetition=rep,
            show_plot=False,
            model=dynamics_model
        )
    generate_GIF(
        experiment=exp_idx, 
        sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
        repetition=rep
        )