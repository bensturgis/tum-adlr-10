import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset
import typing

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
        
        R = 0
        for _ in range(num_particles):
            particle_R = 0
            for a in action_seq:
                _, _, terminated, truncated, info = env.step(a)
                # Accumulate entropy for the current particle
                particle_R += (1. / num_particles) * self.differential_entropy(info["var"])
                if terminated or truncated:
                    break
            # Add the normalized reward for the particle
            R += (1. / num_particles) * particle_R 
            # Reset to the initial state for each particle
            env.unwrapped.state = start_state

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
        _, _ = true_env.reset()

        for t in range(self.horizon):
            # Sample K random action sequences, each of length H
            action_seqs = np.array([
                [learned_env.action_space.sample() for _ in range(self.mpc_horizon)]
                for _ in range(self.num_action_seq)
            ])

            # Evaluate the performance of each action sequence
            performances = np.zeros(self.num_action_seq)
            for k, action_seq in enumerate(action_seqs):
                performances[k] = self.action_seq_performance(learned_env, action_seq, self.num_particles)
                if (k+1) % 50 == 0:
                    print(f"Evaluated performance of {k+1} out of {self.num_action_seq} randomly sampled action sequences.")

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

    