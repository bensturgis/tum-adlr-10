from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import platform
import ctypes
import torch
from typing import Dict, Tuple, Any, Union

from models.feedforward_nn import FeedforwardNN
from models.mc_dropout_bnn import MCDropoutBNN
from models.laplace_bnn import LaplaceBNN

class MassSpringDamperEnv(gym.Env, ABC):
    """
    Abstract base class defining the structure for implementing a Gym-compatible
    mass-spring-damper system.
    """
    def __init__(self) -> None:
        """
        Initialize the mass-spring-damper environment.
        """
        super().__init__()
        self.name = "Mass-Spring-Damper System"

        # State: [position, velocity]
        self.state = np.zeros(2, dtype=np.float32)

        # Action space: Force input constrained to [-input_limit, input_limit]
        force_limit = 1.0
        self.action_space = spaces.Box(
            low=-force_limit,
            high=force_limit,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space: State variables [position, velocity] with unbounded range
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Parameters for rendering environment with pygame
        self.screen = None
        self.window_width, self.window_height = 600, 200
        self.initial_mass_pos = self.window_width // 2 # Initial position of the mass
        self.spring_origin = self.initial_mass_pos - 200 # Fixed anchor point of spring

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        pass

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Included for compatibility with Stable-Baselines3.
            options (Dict[str, Any]): Included for compatibility with Stable-Baselines3.

        Returns:
            Tuple[np.ndarray, dict]: Initial state and additional reset info.
        """
        self.state = np.zeros(2, dtype=np.float32)
        return self.state, {}
    
    def render(self) -> None:
        """
        Renders the current state of the mass-spring-damper system using Pygame.
        """
        # Initialize the pygame screen if not already set
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption(self.name)

        self.screen.fill((255, 255, 255))

        # Calculate the mass's position on the screen
        mass_pos = int(self.initial_mass_pos + self.state[0] * 25)

        # Draw the spring
        pygame.draw.line(self.screen, (0, 0, 0), 
                         (self.spring_origin, self.window_height // 2 - 20),
                         (self.spring_origin, self.window_height // 2 + 20),
                         4)
        pygame.draw.line(self.screen, (0, 0, 0),
                         (self.spring_origin, self.window_height // 2),
                         (mass_pos, self.window_height // 2), 
                         2)

        # Draw the mass
        mass_rect = pygame.Rect(mass_pos - 10, self.height // 2 - 10, 20, 20)
        pygame.draw.rect(self.screen, (0, 255, 0), mass_rect)

        pygame.display.flip()

    def close(self) -> None:
        """
        Closes the rendering window and releases pygame resources.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def compute_state_bounds(self, horizon: int) -> Dict[str, float]:
        """
        Computes the minimum and maximum position and velocity that can be reached
        within the given horizon by always applying either the maximum or minimum action.

        Args:
            horizon (int): Number of steps to simulate forward.

        Returns:
            Dict[str, float]: A dictionary containing the min/max position and velocity keys:
                            {
                                "max_position": float,
                                "min_position": float,
                                "max_velocity": float,
                                "min_velocity": float
                            }
        """
        original_state = self.state.copy()

        # Helper function to simulate and track min/max states using a constant action
        def simulate(action_val):
            self.reset()
            min_pos = self.state[0]
            max_pos = self.state[0]
            min_vel = self.state[1]
            max_vel = self.state[1]

            action = np.array([action_val])
            for _ in range(horizon):
                next_state, _, _, _, _ = self.step(action)
                # Update min/max bounds
                min_pos = min(min_pos, next_state[0])
                max_pos = max(max_pos, next_state[0])
                min_vel = min(min_vel, next_state[1])
                max_vel = max(max_vel, next_state[1])
            
            return min_pos, max_pos, min_vel, max_vel

        # Simulate with minimum action
        min_action = self.action_space.low[0]
        min_pos_min_act, max_pos_min_act, min_vel_min_act, max_vel_min_act = simulate(min_action)

        # Simulate with maximum action
        max_action = self.action_space.high[0]
        min_pos_max_act, max_pos_max_act, min_vel_max_act, max_vel_max_act = simulate(max_action)

        # Combine results
        overall_min_pos = min(min_pos_min_act, min_pos_max_act)
        overall_max_pos = max(max_pos_min_act, max_pos_max_act)
        overall_min_vel = min(min_vel_min_act, min_vel_max_act)
        overall_max_vel = max(max_vel_min_act, max_vel_max_act)

        # Restore the original state
        self.state = original_state

        return {
            "max_position": overall_max_pos,
            "min_position": overall_min_pos,
            "max_velocity": overall_max_vel,
            "min_velocity": overall_min_vel,
            "max_action": max_action,
            "min_action": min_action,
        }
    
    @abstractmethod
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        pass

class TrueMassSpringDamperEnv(MassSpringDamperEnv):
    """
    The "true" mass-spring-damper environment, simulating the real system dynamics.

    This environment models the physical mass-spring-damper system with exact
    parameters, serving as the ground truth for comparison with the learned environment.
    """
    def __init__(
        self, mass: float = 0.1, stiffness: float = 1.0, damping: float = 0.1,
        time_step: float = 0.01, nonlinear: bool = False, noise_var: float = 0.0,
    ) -> None:
        """
        Initialize the "true" mass-spring-damper environment.

        Args:
            mass (float): Mass of the system.
            stiffness (float): Spring constant.
            damping (float): Damping coefficient.
            time_step (float): Discretization time step.
            nonlinear (bool): Whether to use a non-linear stiffness model.
            noise_var (float): Variance of Gaussian noise added to the state.
        """
        super().__init__()
        
        # Physical parameters
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.time_step = time_step
        self.nonlinear = nonlinear
        self.noise_var = noise_var

    def step(
        self, action: np.array
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform single step in the mass-spring-damper environment by applying given action
        according to the real system dynamics.
        
        Args:
            action (np.ndarray): Force applied to the system.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: Next state, reward, termination flag,
                truncation flag, and additional info.
        """
        # Unpack current state
        position, velocity = self.state

        force = action.squeeze()
        if self.nonlinear:
            # Non-linear stiffness model
            spring_force = np.tanh(position)
            new_position = position + self.time_step * velocity
            new_velocity = velocity + self.time_step * (-self.damping * velocity - spring_force + force) / self.mass
            self.state = np.array([new_position, new_velocity], dtype=np.float32)
        else:
            # Linear stiffness model          
            A = np.array([[0, 1], [-self.stiffness / self.mass, -self.damping / self.mass]], dtype=np.float32)
            B = np.array([0, 1 / self.mass], dtype=np.float32)
            self.state = (np.eye(2) + self.time_step * A) @ self.state + self.time_step * B * force

        # Add Gaussian noise to the state
        self.state = np.random.normal(loc=self.state, scale=self.noise_var, size=(2,)).astype(np.float32)

        # Reward is not needed for the true system
        reward = 0.0

        # No termination and truncation conditions defined
        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}
    
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "mass": self.mass,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "time_step": self.time_step,
            "nonlinear": self.nonlinear,
            "noise_var": self.noise_var,
        }
        return parameter_dict

class LearnedMassSpringDamperEnv(MassSpringDamperEnv):
    """
    A mass-spring-damper environment with a learned dynamics model.
    
    This environment simulates a system with mass-spring-damper dynamics
    and uses a learned model (e.g., Bayesian Neural Network or Feedforward Neural Network)
    to predict the next state and reward.
    """
    def __init__(self, model: Union[MCDropoutBNN, FeedforwardNN]):
        """
        Initialize the environment with a learned model.

        Args:
            model (Union[MCDropoutBNN, FeedforwardNN]): The dynamics model to be used
                for state prediction. Can be either a Bayesian Neural Network
                or a standard Feedforward Neural Network.
        """
        super().__init__()
        self.model = model

    def step(self, action: np.array) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform single step in the mass-spring-damper environment by applying given action
        according to learned dynamics.
        
        Args:
            action (np.ndarray): Force applied to the system.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: Next state, reward, termination flag,
                truncation flag, and additional info.
        """
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 2]
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)    # Shape: [1, 1]

        # Model-specific prediction logic
        if isinstance(self.model, MCDropoutBNN) or isinstance(self.model, LaplaceBNN):
            # Bayesian model prediction: returns next state and variance
            next_state, pred_var = self.model.bayesian_pred(state_tensor, action_tensor)
            # Update state
            self.state = next_state.squeeze()
            info = {"var": pred_var.squeeze()}
            reward = self.differential_entropy(pred_var.squeeze())
        elif isinstance(self.model, FeedforwardNN):
            # Deterministic model prediction: only returns next state
            next_state = self.model(state_tensor, action_tensor).squeeze(0).detach().numpy()
            # Update state
            self.state = next_state.squeeze(0).detach().numpy()
            reward = 0.0  # No reward for deterministic models
            info = {}
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}. "
                             f"Expected MCDropoutBNN or FeedforwardNN.")

        # No termination and truncation conditions defined
        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, info
    
    def differential_entropy(self, pred_vars: np.ndarray) -> float:
        """
        Compute the differential entropy of a multivariate Gaussian based on variances from
        bayesian inference.
        
        Args:
            pred_vars (np.ndarray): Predicted variances from bayesian inference of shape [state_dim].
                                    
        Returns:
            float: The computed differential entropy.
        """
        
        # Extract state dimension
        state_dim = pred_vars.size

        # Constant term for Gaussian differential entropy
        const = state_dim * np.log(np.sqrt(2 * np.pi * np.e))

        # Differential entropy computation
        diff_entropy = const + 0.5 * np.sum(np.log(pred_vars))

        return diff_entropy
            
    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "model": self.model.params_to_dict()
        }
        return parameter_dict