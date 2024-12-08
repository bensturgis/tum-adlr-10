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

    def sample_informative_action_seq(self, learned_env: gym.Env, action_seq_length: int,
                                      batch_size: int = 20) -> np.ndarray:
        """
        Generate random action sequences from the specified environment, evaluate their performance and choose
        the most informative action sequence.

        Args:
            learned_env: The environment from which to sample and evaluate actions.
            action_seq_length (int): The number of steps in each action sequence. Corresponds to MPC horizon (H) if
                                     we use RS + MPC and to horizon (T) if we use RS without MPC.

        Returns:
            np.ndarray: Array of length `action_seq_length` containing the most informative action sequence of the
                        `num_action_seq` sampled action sequences.
        """
        device = next(learned_env.unwrapped.model.parameters()).device
        state_dim = learned_env.observation_space.shape[0]
        action_dim = learned_env.action_space.shape[0]
        
        # Sample K random action sequences, each of length `action_seq_length`
        action_seqs = np.random.uniform(
            low=learned_env.action_space.low,
            high=learned_env.action_space.high,
            size=(self.num_action_seq, action_seq_length, action_dim)
        )

        # Store the computed performances for each sequence
        performances = np.zeros(self.num_action_seq)

        # Process in chunks of batch_size
        for start_idx in tqdm(range(0, self.num_action_seq, batch_size), desc="Evaluating action sequences"):
            end_idx = min(start_idx + batch_size, self.num_action_seq)
            batch_action_seqs = action_seqs[start_idx:end_idx]  # shape: [batch_size, action_seq_length, action_dim]
            current_batch_size = end_idx - start_idx

            # Allocate tensors for states and actions for this batch
            states = torch.zeros([current_batch_size, self.num_particles, action_seq_length, state_dim], device=device)
            start_state = torch.from_numpy(learned_env.unwrapped.state).float().to(device)
            start_state = start_state.unsqueeze(0).unsqueeze(0)  # [1, 1, Ds]
            start_state = start_state.repeat(current_batch_size, self.num_particles, 1)  # [batch_size, P, Ds]
            states[:, :, 0] = start_state

            actions = torch.from_numpy(batch_action_seqs).float().to(device)  # [batch_size, H, Da]
            actions = actions.unsqueeze(1).repeat(1, self.num_particles, 1, 1)  # [batch_size, P, H, Da]

            # Compute next states for each timestep
            for k in range(action_seq_length - 1):
                next_states = learned_env.unwrapped.model(
                    states[:, :, k].view(current_batch_size * self.num_particles, state_dim),
                    actions[:, :, k].view(current_batch_size * self.num_particles, action_dim)
                )
                states[:, :, k + 1] = next_states.view(current_batch_size, self.num_particles, state_dim)

            # Bayesian prediction over all steps in the batch
            states_batch = states.view(current_batch_size * self.num_particles * action_seq_length, state_dim)
            actions_batch = actions.view(current_batch_size * self.num_particles * action_seq_length, action_dim)
            _, variance_values = learned_env.unwrapped.model.bayesian_pred(states_batch, actions_batch)
            variance_values = variance_values.reshape(current_batch_size, self.num_particles, action_seq_length, state_dim)

            const = action_seq_length * state_dim * np.log(np.sqrt(2 * np.pi * np.e))
            diff_entrop_list = const + 0.5 * np.sum(np.log(variance_values), axis=(2, 3))  # [batch_size, num_particles]
            batch_performances = np.mean(diff_entrop_list, axis=1)  # [batch_size]

            # Store performances for this batch
            performances[start_idx:end_idx] = batch_performances

        print(f"Evaluated performance of {self.num_action_seq} randomly sampled action sequences.")

        # Choose the best action sequence (highest performance)
        best_action_seq = action_seqs[np.argmax(performances)]

        return best_action_seq

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