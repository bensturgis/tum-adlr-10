import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset
import typing
from tqdm import tqdm

from sampling_methods.sampling_method import SamplingMethod

class RandomSamplingShooting(SamplingMethod):
    def __init__(
            self, horizon: int, mpc_horizon: int, num_action_seq: int, 
            num_chosen_action_seq: int, num_particles: int
        ) -> None:
        """
        Initialize RandomSamplingShooting object.

        Args:
            horizon (T) (int): The total planning horizon (T).
            mpc_horizon (H) (int): The number of steps (H) in each action sequence.
            num_action_seq (K) (int): The number of action sequences (K) to sample at each 
                                      time step.
            num_particles (P) (int): The number of particles for Monte Carlo sampling during
                                     performance evaluation.
        """
        super().__init__(horizon=horizon)
        self.mpc_horizon = mpc_horizon
        self.num_action_seq = num_action_seq
        self.num_chosen_action_seq = num_chosen_action_seq
        self.num_particles = num_particles
    
    def differential_entropy(self, pred_var: np.ndarray) -> float:
        """
        Computes differential entropy for a multivariate Gaussian.

        This method implements Equation 7 from the paper 
        "Actively learning dynamical systems using Bayesian neural networks."

        Args:
            pred_var (\sigma^2) (np.ndarray): An array of variances for each dimension of the
                                              Bayesian prediction.
                
        Returns:
            float: The computed differential entropy value \mathcal{H}[f(s_t, a_t)].
        """
        d_s = pred_var.shape[0]
        const = d_s * np.log(np.sqrt(2 * np.pi * np.exp(1)))
        diff_entrop = const + (1. / 2.) * np.sum(np.log(pred_var))
        return diff_entrop

    def action_seq_performance(self, env: gym.Env, action_seq: np.array, num_particles: int) -> float:
        """
        Computes the performance of a sequence of actions using Monte Carlo
        integration to approximate the reward function R(s_t, a_{t:t+H-1}).

        This method implements Equation 13 from the paper 
        "Actively learning dynamical systems using Bayesian neural networks."

        Args:
            env (gym.Env): The environment to evaluate the action sequence on.
            action_seq (a_{t:t+H-1}) (list): A list of actions to be executed.
            num_particles (P) (int): The number of particles (P) for Monte Carlo integration.

        Returns:
            float: The approximated reward value R(s_t, a_{t:t+H-1}).
        """
        # Save the initial state for the particle trajectories
        start_state = env.unwrapped.state
        
        # record (s,a) sequences from P trajectories
        H = len(action_seq) # horizon
        Ds = env.observation_space.shape[0] # state dimension
        Da = env.action_space.shape[0] # action dimension
        device = next(env.unwrapped.model.parameters()).device
        states = torch.zeros([num_particles, H, Ds], device=device) # states record [P, H, Ds]
        actions = torch.zeros([num_particles, H, Da], device=device) # actions record [P, H, Da]
        states[:, 0] = torch.from_numpy(start_state).to(device)
        actions[:, :] = torch.from_numpy(action_seq).to(device)
        for k in range(H-1): # parallelize P trajectory collection
            states[:, k+1] = env.unwrapped.model(states[:, k], actions[:, k])
        states_batch = states.view(num_particles*H, Ds)
        actions_batch = actions.view(num_particles*H, Da)
        _, var_list = env.unwrapped.model.bayesian_pred(states_batch, actions_batch)
        # data in var_list(np.ndarray): along one trajectory(H) and then particles(P)
        vars = var_list.reshape(num_particles, H, Ds)
        # directly calc diff_entropy of trajectories
        const = H * Ds * np.log(np.sqrt(2 * np.pi * np.e))
        diff_entrop_list = const + (1. / 2.) * np.sum(np.log(vars), axis=(1,2)) # directly sum over [H, Ds] to get accumulated entropy for a traj
        R = np.mean(diff_entrop_list) # average on particle dimension

        return R
                
    def sample(self, true_env: gym.Env, learned_env: gym.Env) -> TensorDataset:
        """
        Implements Random Shooting Model Predictive Control (RS+MPC) as described in Algorithm 4
        from the paper "Actively learning dynamical systems using Bayesian neural networks."

        Args:
            true_env (gym.Env): The true environment for the actual data collection.
            learned_env (gym.Env): The learned environment using a dynamics model to find the most
                                   informative action sequence.

        Returns:
            TensorDataset: A PyTorch TensorDataset containing the collected
                        (state, action, next_state) transitions.
        """
        # Lists to store states, actions, and next states collected during sampling
        states = []
        actions = []
        next_states = []

        # Reset the environment to initial state
        true_env.reset()

        if self.mpc_horizon == 0: # RS without MPC
            action_seqs = np.random.uniform(low=learned_env.action_space.low, high=learned_env.action_space.high, size=(self.num_action_seq, self.horizon, 1))
            # Evaluate the performance of each action sequence
            performances = np.zeros(self.num_action_seq)
            for k, action_seq in enumerate(tqdm(action_seqs, desc="Evaluating action sequences")):
                performances[k] = self.action_seq_performance(learned_env, action_seq, self.num_particles)
            chosen_action_seqs_indices = np.argsort(performances)[::-1][:self.num_chosen_action_seq]
            print(f"Chosen {self.num_chosen_action_seq} sequences of {self.num_action_seq} randomly sampled action sequences.")
            for n in range(self.num_chosen_action_seq):
                action_seq = action_seqs[chosen_action_seqs_indices[n]]
                true_env.reset()
                for t in range(self.horizon):
                    # Append the current state to the list
                    states.append(true_env.unwrapped.state)

                    # Append the t-th action of the action sequence to the list
                    actions.append(action_seq[t])

                    # Step the environment using the sampled action and append the next state
                    next_state, _, terminated, truncated, _ = true_env.step(action_seq[t])
                    next_states.append(next_state)

                    # Reset if the environment reaches a terminal or truncated state
                    if terminated or truncated:
                        true_env.reset()

            print(f"Collected {self.num_chosen_action_seq*self.horizon} samples from real environment")  
        
        else: # RS + MPC

            for t in range(self.horizon):
                # Sample K random action sequences, each of length H
                action_seqs = np.random.uniform(low=learned_env.action_space.low, high=learned_env.action_space.high, size=(self.num_action_seq, self.mpc_horizon, 1))

                # Evaluate the performance of each action sequence
                performances = np.zeros(self.num_action_seq)
                for k, action_seq in enumerate(action_seqs):
                    learned_env.reset(state=true_env.unwrapped.state)
                    performances[k] = self.action_seq_performance(learned_env, action_seq, self.num_particles)
                print(f"Evaluated performance of {self.num_action_seq} randomly sampled action sequences.")

                # Choose the best action sequence (highest performance)
                best_action_seq = action_seqs[np.argmax(performances)]

                # Append the current state to the list
                states.append(true_env.unwrapped.state)

                # Append the first action of the best performing action sequence to the list
                actions.append(best_action_seq[0])

                # Step the environment using the sampled action and append the next state
                next_state, _, terminated, truncated, _ = true_env.step(best_action_seq[0])
                next_states.append(next_state)

                # Reset if the environment reaches a terminal or truncated state
                if terminated or truncated:
                    true_env.reset()

                print(f"Collected {t+1} actions for action sequence of horizon {self.horizon}.")

        # Convert the collected lists into tensors
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        action_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
        next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)

        return TensorDataset(state_tensor, action_tensor, next_state_tensor)

    