import sys
import os
sys.path.append(os.path.abspath("."))

from metrics.one_step_pred_accuracy import OneStepPredictionErrorEvaluator
from metrics.multi_step_pred_accuracy import MultiStepPredictionErrorEvaluator
from utils.visualization_utils import get_horizon, plot_prediction_error, reconstruct_envs

EXPERIMENT = 2
# EVAL_REPETITIONS = [0, 1, 2, 3, 4] # for reacher (experiment 1)
EVAL_REPETITIONS = [0, 1, 2, 3] # for mass-spring-damper system (experiment 2)
PLOT_VARIANCES = True
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

# Initialize the evluation metrics 
one_step_prediction_error = OneStepPredictionErrorEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    num_samples=NUM_SAMPLES,
    horizon=get_horizon(experiment=EXPERIMENT)
)
multi_step_pred_acc_eval = MultiStepPredictionErrorEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    horizon=get_horizon(experiment=EXPERIMENT),
    num_trajectories=NUM_TRAJECTORIES,
    trajectory_horizon=TRAJCETORY_LENGTH,
    num_initial_states=NUM_INITIAL_STATES,
    num_prediction_steps=NUM_PREDICTION_STEPS,
)

plot_prediction_error(
    experiment=EXPERIMENT,
    learned_env=learned_env,
    metrics=[one_step_prediction_error],
    eval_repetitions=[0, 1, 2, 3],
    plot_variances=PLOT_VARIANCES
)