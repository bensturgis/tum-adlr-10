import sys
import os
sys.path.append(os.path.abspath("."))

from environments.mass_spring_damper_system import TrueMassSpringDamperEnv, LearnedMassSpringDamperEnv
from models.mc_dropout_bnn import MCDropoutBNN

from stable_baselines3.common.env_checker import check_env

# Hyperparameters for neural network and training
HIDDEN_SIZE = 72          # Hidden units in the neural network
DEVICE = "cpu"           # PyTorch device for training
DROP_PROB = 0.1           # Dropout probability for the bayesian neural network

# Initialize the true environment
true_env = TrueMassSpringDamperEnv()
# Check whether true environment is compatbile with Stable-Baselines3
check_env(true_env, warn=True)

# Set up the dynamics model and learned environment
state_dim = true_env.state_dim
action_dim = true_env.action_dim
# dynamics_model = FeedforwardNN(state_dim, action_dim, hidden_size=HIDDEN_SIZE)
dynamics_model = MCDropoutBNN(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_size=HIDDEN_SIZE,
    drop_prob=DROP_PROB,
    device=DEVICE,
)
learned_env = LearnedMassSpringDamperEnv(model=dynamics_model)
# Check whether learned environment is compatbile with Stable-Baselines3
check_env(learned_env, warn=True)