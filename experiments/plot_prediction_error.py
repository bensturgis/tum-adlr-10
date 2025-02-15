import sys
import os
sys.path.append(os.path.abspath("."))

from metrics.one_step_pred_accuracy import OneStepPredictionErrorEvaluator
from utils.visualization_utils import get_horizon, plot_prediction_error, reconstruct_envs

EXPERIMENT = 1
DEVICE = "cpu"            # PyTorch device for evaluation

# General hyperparamters for prediction error
BOUND_SHRINK_FACTOR = 1.0

# Hyperparameters for one-step prediction error
NUM_SAMPLES = 1250        # Number of (state, action, next_state) samples (N_1 = 1250 in paper)

# Hyperparameters for multi-step prediction error
NUM_TRAJECTORIES = 125     # Number of full trajectories generated in the true environment (N_2 = 10 in paper)
TRAJCETORY_LENGTH = 50   # Maximum length of each trajectory (not specified in paper)
NUM_INITIAL_STATES = 10   # Number of initial states sampled from each trajectory (not specified in paper)
NUM_PREDICTION_STEPS = 20 # Number of steps for multi-step prediction evaluation (M = 20 in paper)

# TODO: assert that environments match
true_env, learned_env = reconstruct_envs(experiment=EXPERIMENT, device=DEVICE)

true_env.set_bound_shrink_factor(bound_shrink_factor=BOUND_SHRINK_FACTOR)

one_step_prediction_error = OneStepPredictionErrorEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    num_samples=NUM_SAMPLES,
    horizon=get_horizon(experiment=EXPERIMENT)
)

plot_prediction_error(
    experiment=EXPERIMENT,
    learned_env=learned_env,
    metrics=[one_step_prediction_error],
)