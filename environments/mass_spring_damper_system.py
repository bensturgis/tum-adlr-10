from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import torch
from typing import Any, Dict, List, Tuple, Union

from environments.learned_env import LearnedEnv
from models.feedforward_nn import FeedforwardNN
from models.bnn import BNN
from utils.train_utils import compute_state_bounds

class MassSpringDamperEnv(gym.Env, ABC):
    """
    Abstract base class defining the structure for implementing a Gym-compatible
    mass-spring-damper system.
    """
    def __init__(self) -> None:
        """
        Initialize the mass-spring-damper environment.
        """
        gym.Env.__init__(self)
        self.name = "Mass-Spring-Damper System"

        # State: [position, velocity]
        self.state = np.zeros(2, dtype=np.float32)
        self.state_dim = 2

        # Define the state dimension names
        self.state_dim_names = {
            0: "position",
            1: "velocity"
        }

        # Action space: Force input constrained to [-input_limit, input_limit]
        force_limit = 1.0
        self.action_space = spaces.Box(
            low=-force_limit,
            high=force_limit,
            shape=(1,),
            dtype=np.float32
        )
        self.action_dim = 1

        # Observation space: State variables [position, velocity] with unbounded range
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Ensures the input magnitude to the Bayesian neural network remains constant
        self.input_expansion = True

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

    def get_state_bounds(self, horizon: int) -> Dict[int, np.array]:
        """
        Computes and retrieves state bounds over the specified horizon.

        Args:
            horizon (int): Number of simulation steps to determine bounds.

        Returns:
            Dict[int, np.ndarray]: Mapping of state dimension index to their [min, max] bounds.
        """
        return compute_state_bounds(env=self, horizon=horizon)
    
    def get_action_bounds(self) -> Dict[int, np.array]:
        """
        Retrieves the action bounds for each action dimension.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping action dimension index to their ,
                                   [min, max] bounds.
        """
        # Extract action bounds for all dimensions
        action_bounds = {}
        for dim_idx in range(self.action_dim):
            action_bounds[dim_idx] = np.array(
                [self.action_space.low[dim_idx], self.action_space.high[dim_idx]],
                np.float32
            )

        return action_bounds
    
    def define_sampling_bounds(
        self, horizon: int, bound_shrink_factor: float = 0.5
    ) -> Dict[int, np.array]:
        """
        Defines bounds to sample data for metrics that evaluate performance of the
        learned environment.

        Args:
            horizon (int): Number of steps to simulate for each action.
            bound_shrink_factor (float): Factor by which to shrink maximum/minimum state bounds.

        Returns:
            Dict[int, np.array]: Dictionary mapping state dimension index to their
                                 sampling bounds.
        """
        # Compute the raw minimum/maximum state bounds
        state_bounds = self.get_state_bounds(horizon=horizon)
        
        sampling_bounds = {}        
        for dim_idx in range(self.state_dim):
            # Apply the shrink factor to the minimum/maximum state bounds
            sampling_bounds[dim_idx] = bound_shrink_factor * state_bounds[dim_idx]
        
        return sampling_bounds
    
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

class LearnedMassSpringDamperEnv(MassSpringDamperEnv, LearnedEnv):
    """
    A mass-spring-damper environment with a learned dynamics model.
    """
    def __init__(self, model: Union[FeedforwardNN, BNN]) -> None:
        """
        Initialize "learned" mass-spring-damper system.
        """
        LearnedEnv.__init__(self, model=model)
        MassSpringDamperEnv.__init__(self)
        

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return LearnedEnv.step(self, action)
    
    def params_to_dict(self) -> Dict[str, str]:
        return LearnedEnv.params_to_dict(self)
        
    