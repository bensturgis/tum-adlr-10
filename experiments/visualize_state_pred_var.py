import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath("."))

# test model
from dynamical_systems.mass_spring_damper_system import MassSpringDamperEnv
from models.mc_dropout_bnn import MCDropoutBNN
from utils.visualization_utils import plot_state_space_pred_var
import numpy as np
import torch
import matplotlib.pyplot as plt

# Hyperparameters for neural network and training
HIDDEN_SIZE = 64          # Hidden units in the neural network
NUM_EPOCHS = 25           # Training epochs per iteration
BATCH_SIZE = 50           # Batch size for training
LEARNING_RATE = 1e-3      # Learning rate for the optimizer
DEVICE = "cuda"           # PyTorch device for training
DROP_PROB = 0.1           # Dropout probability for the bayesian neural network

env = MassSpringDamperEnv(nlin=True)
# Initialize trained dynamics model
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
trained_model = MCDropoutBNN(state_dim, action_dim, hidden_size=HIDDEN_SIZE, drop_prob=DROP_PROB)

# load saved weights
trained_model.load_state_dict(torch.load('./weights/mc_dropout_mass_spring_damper.pth'))

plot_state_space_pred_var(trained_model, [[-3,3], [-3,3]], [[-1,1]])