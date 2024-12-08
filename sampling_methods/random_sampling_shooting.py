import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Tuple
from tqdm import tqdm

from sampling_methods.sampling_method import SamplingMethod

class RandomSamplingShooting(SamplingMethod):
    def __init__(
            self, horizon: int, mpc_horizon: int, num_action_seq: int, 
            num_particles: int
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
        self.num_particles = num_particles
        self.name = "Random Sampling Shooting"

    def compute_performances(self, pred_vars: np.ndarray) -> np.ndarray:
        """
        Computes performance scores for a batch action sequences. The performance
        is derived from the differential entropy of the multivariate Guassians of each
        bayesian prediction along the trajectory and averaged over a number of particles
        for monte carlo integration. 

        Args:
            pred_var (np.ndarray): Variances of bayesian prediction of shape
                                   [batch_size, num_particles, action_seq_length, state_dim] where
                                   `num_particles` is denoted as P and state_dim is denoted as d_s
                                   in the paper. `action_seq_length` can either be the MPC horizon H
                                   if we use RS+MPC or the entire horizon T if we use RS without MPC.

        Returns:
            np.ndarray: Performance scores of shape [batch_size], one score for each action sequence
                        in the batch.
        """
        # Extract dimensions
        _, _, action_seq_length, state_dim = pred_vars.shape

        # Constant term for Gaussian differential entropy
        const = action_seq_length * state_dim * np.log(np.sqrt(2 * np.pi * np.e))

        # Differential entropy for each particle of each action sequence
        diff_entrop = const + 0.5 * np.sum(np.log(pred_vars), axis=(2, 3)) # shape: [batch_size, num_particles]

        # Monte carlo integration of performance score by averaging over the particles
        performances = np.mean(diff_entrop, axis=1) # shape: [batch_size]

        return performances


    def sample_informative_action_seq(self, learned_env: gym.Env, action_seq_length: int,
                                      batch_size: int = 20) -> np.ndarray:
        """
        Generate random action sequences from the specified environment, evaluate their performance
        and choose the most informative one based on differential entropy.

        Args:
            learned_env (gym.Env): The environment with a learned dynamics model.
            action_seq_length (int): The length of each action sequence (MPC horizon H for random
                                     sampling shooting combined with MPC and complete horizon T for
                                     random sampling shooting without MPC).
            batch_size (int): The number of action sequences to evaluate simultaneously in a single batch.
                              Defaults to 20.
            
        Returns:
            np.ndarray: The most informative action sequence of shape `[action_seq_length, action_dim]`.
        """
        # Determine the device used for computations related to learned model
        device = next(learned_env.unwrapped.model.parameters()).device
        
        # Extract state and action dimensions from the environment
        state_dim = learned_env.observation_space.shape[0]
        action_dim = learned_env.action_space.shape[0]
        
        # Sample `self.num_action_seq` random action sequences from the action space
        action_seqs = np.random.uniform(
            low=learned_env.action_space.low,
            high=learned_env.action_space.high,
            size=(self.num_action_seq, action_seq_length, action_dim)
        ) # Shape: [num_action_seq, action_seq_length, action_dim]

        # Store the performances for each action sequence
        performances = np.zeros(self.num_action_seq)

        # Evaluate action sequences in batches
        for start_idx in tqdm(range(0, self.num_action_seq, batch_size), desc="Evaluating action sequences"):
            end_idx = min(start_idx + batch_size, self.num_action_seq)
            current_batch_size = end_idx - start_idx

            # Extract the current batch of action sequences
            batch_action_seqs = action_seqs[start_idx:end_idx]  # shape: [current_batch_size, action_seq_length, action_dim]

            # Initialize the states tensor for this batch
            states = torch.zeros([current_batch_size, self.num_particles, action_seq_length, state_dim], device=device)
            # Set the initial state for all particles
            start_state = torch.from_numpy(learned_env.unwrapped.state).float().to(device) # [state_dim]
            start_state = start_state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
            start_state = start_state.repeat(current_batch_size, self.num_particles, 1)  # [batch_size, num_particles, state_dim]
            states[:, :, 0] = start_state

            # Replicate action sequences across particles
            actions = torch.from_numpy(batch_action_seqs).float().to(device)  # [batch_size, H, Da]
            actions = actions.unsqueeze(1).repeat(1, self.num_particles, 1, 1)  # [batch_size, P, H, Da]

            # Compute next states for each timestep
            for k in range(action_seq_length - 1):
                next_states = learned_env.unwrapped.model(
                    states[:, :, k].view(current_batch_size * self.num_particles, state_dim),
                    actions[:, :, k].view(current_batch_size * self.num_particles, action_dim)
                )
                states[:, :, k + 1] = next_states.view(current_batch_size, self.num_particles, state_dim)

            # Bayesian prediction for all (state, action) pairs in the batch
            states_batch = states.view(current_batch_size * self.num_particles * action_seq_length, state_dim)
            actions_batch = actions.view(current_batch_size * self.num_particles * action_seq_length, action_dim)
            _, pred_vars = learned_env.unwrapped.model.bayesian_pred(states_batch, actions_batch)
            pred_vars = pred_vars.reshape(current_batch_size, self.num_particles, action_seq_length, state_dim)

            # Compute and store performances for this batch based on differential entropy
            performances[start_idx:end_idx] = self.compute_performances(pred_vars)

        print(f"Evaluated performance of {self.num_action_seq} randomly sampled action sequences.")

        # Choose action sequence with the highest performance score
        best_action_seq = action_seqs[np.argmax(performances)]

        return best_action_seq

    def sample(self, true_env: gym.Env, learned_env: gym.Env) -> TensorDataset:
        """
        Implements Random Shooting Model Predictive Control (RS+MPC) as described in Algorithm 4
        from the paper "Actively learning dynamical systems using Bayesian neural networks."

        Args:
            true_env (gym.Env): The true environment for the actual data collection.
            learned_env (gym.Env): The environment with a learned dynamics model.

        Returns:
            TensorDataset: A PyTorch TensorDataset containing the collected
                        (state, action, next_state) transitions.
        """
        # Lists to store states, actions, and next states collected during sampling
        states = []
        actions = []
        next_states = []

        # Reset the environments to initial state
        learned_env.reset()
        true_env.reset()

        if self.mpc_horizon == 0: # RS without MPC
            best_action_seq = self.sample_informative_action_seq(
                                     learned_env=learned_env,
                                     action_seq_length=self.horizon
                                   )            
            for t in range(self.horizon):
                # Append the current state to the list
                states.append(true_env.unwrapped.state)

                # Append the t-th action of the best action sequence to the list
                actions.append(best_action_seq[t])

                # Step the environment using the sampled action and append the next state
                next_state, _, terminated, truncated, _ = true_env.step(best_action_seq[t])
                next_states.append(next_state)

                # Reset if the environment reaches a terminal or truncated state
                if terminated or truncated:
                    true_env.reset()

            print(f"Collected {self.horizon} samples from real environment") 

        else: # RS + MPC
            for t in range(self.horizon):
                best_action_seq = self.sample_informative_action_seq(
                                     learned_env=learned_env,
                                     action_seq_length=self.horizon
                                   )    
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