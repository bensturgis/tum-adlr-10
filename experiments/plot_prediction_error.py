import sys
import os
sys.path.append(os.path.abspath("."))

from metrics.one_step_pred_accuracy import OneStepPredictionErrorEvaluator
from utils.visualization_utils import get_horizon, plot_prediction_error, reconstruct_envs

EXPERIMENT = 21
DEVICE = "cpu"            # PyTorch device for evaluation

# Hyperparameters for one-step predictive accuracy
NUM_SAMPLES = 1250        # Number of (state, action, next_state) samples (N_1 = 1250 in paper)


# TODO: assert that environments match
true_env, learned_env = reconstruct_envs(experiment=EXPERIMENT, device=DEVICE)

one_step_prediction_error = OneStepPredictionErrorEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    num_samples=NUM_SAMPLES,
    horizon=get_horizon(experiment=EXPERIMENT)
)

plot_prediction_error(
    experiment=EXPERIMENT,
    learned_env=learned_env,
    metrics=[one_step_prediction_error]
)