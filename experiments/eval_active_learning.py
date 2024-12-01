import sys
import os
sys.path.append(os.path.abspath("."))

# Import modules and classes
from dynamical_systems.mass_spring_damper_system import MassSpringDamperEnv
from models.feedforward_nn import FeedforwardNN
from models.mc_dropout_bnn import MCDropoutBNN
from sampling_methods.random_exploration import RandomExploration
from sampling_methods.random_sampling_shooting import RandomSamplingShooting
from active_learning import ActiveLearningEvaluator

# Hyperparameters for neural network and training
HIDDEN_SIZE = 64          # Hidden units in the neural network
NUM_EPOCHS = 25           # Training epochs per iteration
BATCH_SIZE = 25           # Batch size for training
LEARNING_RATE = 1e-3      # Learning rate for the optimizer
DEVICE = "cuda"           # PyTorch device for training
DROP_PROB = 0.1           # Dropout probability for the bayesian neural network

# General hyperparameters for sampling method
HORIZON = 50              # Trajectory time horizon (T = 50 in paper)

# Hyperparameters for random sampling shooting
MPC_HORIZON = 10 # Number of steps (H) in each sampled action sequence (H = 10 in paper)
NUM_ACTION_SEQ = 20000 # Number of action sequences (K) sampled at each time step (K = 20000 in paper)
NUM_PARTICLES = 100 # The number of particles for Monte Carlo sampling during performance evaluation

# Hyperparameters for the active learning evaluation
NUM_AL_ITERATIONS = 20    # Number of active learning iterations (20 in paper)
NUM_EVAL_REPETITIONS = 10  # Number of evaluation runs for mean and variance (20 in paper)

# Initialize the true environment
true_env = MassSpringDamperEnv()

# Set up the dynamics model and learned environment
state_dim = true_env.observation_space.shape[0]
action_dim = true_env.action_space.shape[0]
# dynamics_model = FeedforwardNN(state_dim, action_dim, hidden_size=HIDDEN_SIZE)
dynamics_model = MCDropoutBNN(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_size=HIDDEN_SIZE,
    drop_prob=DROP_PROB,
    device=DEVICE,
)
learned_env = MassSpringDamperEnv(model=dynamics_model)

# Initialize the sampling method
random_exploration = RandomExploration(horizon=HORIZON)
random_sampling_shooting = RandomSamplingShooting(
    horizon=HORIZON,
    mpc_horizon=MPC_HORIZON,
    num_action_seq=NUM_ACTION_SEQ,
    num_particles=NUM_PARTICLES,
)

# Initialize the active learning evaluator
active_learning_evaluator = ActiveLearningEvaluator(
    true_env=true_env,
    learned_env=learned_env,
    sampling_method=random_sampling_shooting,
    num_al_iterations=NUM_AL_ITERATIONS,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_eval_repetitions=NUM_EVAL_REPETITIONS,
)

# Run the active learning process
active_learning_evaluator.active_learning()