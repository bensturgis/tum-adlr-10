import sys
import os
sys.path.append(os.path.abspath("."))

# Import modules and classes
from active_learning import ActiveLearningEvaluator
from dynamical_systems.mass_spring_damper_system import TrueMassSpringDamperEnv, LearnedMassSpringDamperEnv
from metrics.one_step_pred_accuracy import OneStepPredictiveAccuracyEvaluator
from metrics.multi_step_pred_accuracy import MultiStepPredictiveAccuracyEvaluator
from models.feedforward_nn import FeedforwardNN
from models.mc_dropout_bnn import MCDropoutBNN
from models.laplace_bnn import LaplaceBNN
from sampling_methods.random_exploration import RandomExploration
from sampling_methods.random_sampling_shooting import RandomSamplingShooting
from sampling_methods.soft_actor_critic import SoftActorCritic

# Hyperparameters for neural network and training
HIDDEN_SIZE = 72          # Hidden units in the neural network
NUM_EPOCHS = 25           # Training epochs per iteration
BATCH_SIZE = 50           # Batch size for training
LEARNING_RATE = 1e-3      # Learning rate for the optimizer
DEVICE = "cuda"           # PyTorch device for training
DROP_PROB = 0.1           # Dropout probability for the bayesian neural network

# General hyperparameters for sampling method
HORIZON = 50              # Trajectory time horizon (T = 50 in paper)

# Hyperparameters for random sampling shooting
MPC_HORIZON = 20 # Number of steps (H) in each sampled action sequence (H = 10 in paper) / Set H = 0 to discard MPC
NUM_ACTION_SEQ = 2000 # Number of action sequences (K) sampled at each time step (K = 20000 in paper)
NUM_PARTICLES = 100 # The number of particles for Monte Carlo sampling during performance evaluation

# Hyperparamters for the Soft Actor-Critic (SAC)
TOTAL_TIMESTEPS = 25000 # The total number of timesteps to train the SAC in each active learning iteration

# Hyperparameters for one-step predictive accuracy
NUM_SAMPLES = 1250        # Number of (state, action, next_state) samples (N_1 = 1250 in paper)

# TODO: find suitable hyperparameters to evaluate multi-step predictive accuracy
# Hyperparameters for multi-step predictive accuracy
NUM_TRAJECTORIES = 125     # Number of full trajectories generated in the true environment (N_2 = 10 in paper)
TRAJCETORY_LENGTH = 50   # Maximum length of each trajectory (not specified in paper)
NUM_INITIAL_STATES = 10   # Number of initial states sampled from each trajectory (not specified in paper)
NUM_PREDICTION_STEPS = 20 # Number of steps for multi-step prediction evaluation (M = 20 in paper)

# Hyperparameters for the active learning evaluation
NUM_AL_ITERATIONS = 2    # Number of active learning iterations (20 in paper)
NUM_EVAL_REPETITIONS = 1  # Number of evaluation runs for mean and variance (20 in paper)

# Initialize the true environment
true_env = TrueMassSpringDamperEnv(noise_var=0.0)

# Extract the minimum and maximum state values the environment can reach for the given horizon
state_bounds = true_env.compute_state_bounds(horizon=HORIZON)

# Set up the dynamics model and learned environment
state_dim = true_env.observation_space.shape[0]
action_dim = true_env.action_space.shape[0]
# dynamics_model = FeedforwardNN(state_dim, action_dim, hidden_size=HIDDEN_SIZE)
# dynamics_model = MCDropoutBNN(
#     state_dim=state_dim,
#     action_dim=action_dim,
#     hidden_size=HIDDEN_SIZE,
#     drop_prob=DROP_PROB,
#     device=DEVICE,
# )
dynamics_model = LaplaceBNN(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_size=HIDDEN_SIZE,
    device=DEVICE,
)
learned_env = LearnedMassSpringDamperEnv(model=dynamics_model)

# Initialize the sampling methods
random_exploration = RandomExploration(horizon=HORIZON)
random_sampling_shooting = RandomSamplingShooting(
    horizon=HORIZON,
    mpc_horizon=MPC_HORIZON,
    num_action_seq=NUM_ACTION_SEQ,
    num_particles=NUM_PARTICLES,
)
soft_actor_critic = SoftActorCritic(
    horizon=HORIZON, 
    total_timesteps=TOTAL_TIMESTEPS
)
sampling_methods = [random_sampling_shooting]

# Initialize the evluation metrics 
one_step_pred_acc_eval = OneStepPredictiveAccuracyEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    state_bounds=state_bounds,
    num_samples=NUM_SAMPLES,
)
    
multi_step_pred_acc_eval = MultiStepPredictiveAccuracyEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    state_bounds=state_bounds,
    num_trajectories=NUM_TRAJECTORIES,
    trajectory_horizon=TRAJCETORY_LENGTH,
    num_initial_states=NUM_INITIAL_STATES,
    num_prediction_steps=NUM_PREDICTION_STEPS,
)
evaluation_metrics = [one_step_pred_acc_eval]

# Initialize the active learning evaluator
active_learning_evaluator = ActiveLearningEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    sampling_methods=sampling_methods,
    evaluation_metrics=evaluation_metrics,
    num_al_iterations=NUM_AL_ITERATIONS,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_eval_repetitions=NUM_EVAL_REPETITIONS,
)

# Run the active learning process
active_learning_evaluator.active_learning()