from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Any, Dict, Tuple, Union

from environments.learned_env import LearnedEnv
from models.feedforward_nn import FeedforwardNN
from models.bnn import BNN
from utils.train_utils import compute_state_bounds

class ReacherEnv(gym.Env, ABC):
    """
    Abstract base class for a 2-joint Reacher environment whose state is given by
    s = [cos(theta1), cos(theta2), sin(theta1), sin(theta2), vx, vy].
    """
    def __init__(self, link_length: float = 0.5) -> None:
        """
        Initialize the Reacher environment.

        Args:
            link_length (float): Length of each link.
        """
        gym.Env.__init__(self)
        self.name = "Reacher"

        # Environment uses a 6D state as described:
        # [cos(theta1), cos(theta2), sin(theta1), sin(theta2), vx, vy]
        self.state = np.zeros(6, dtype=np.float32)
        self.state_dim = 6

        # Define the state dimension names
        self.state_dim_names = {
            0: "cos(theta1)",
            1: "cos(theta2)",
            2: "sin(theta1)",
            3: "sin(theta2)",
            4: "velocity_x",
            5: "velocity_y"
        }

        # Two torques, each in [-0.2, 0.2]
        torque_limit = 0.2
        self.action_space = spaces.Box(
            low=-torque_limit,
            high=torque_limit,
            shape=(2,),
            dtype=np.float32
        )
        self.action_dim = 2

        # cos/sin will be in [-1,1] and vx, vy are unbounded
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0,  1.0,  np.inf,  np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Input expansion disabled since the magnitude of the state remains approximately constant
        self.input_expansion = False

        # Rendering
        self.screen = None
        self.window_width, self.window_height = 500, 500

        # Link lengths
        self.link_length = link_length

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict]:
        pass

    def render(self) -> None:
        """
        Renders the current state of the reacher using Pygame.
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption(self.name)

        self.screen.fill((255, 255, 255))

        # Convert cos/sin to angles
        cos_t1, cos_t2, sin_t1, sin_t2, _, _ = self.state
        theta1 = np.arctan2(sin_t1, cos_t1)
        theta2 = np.arctan2(sin_t2, cos_t2)

        origin = (self.window_width // 2, self.window_height // 2)
        scale = 100

        # Coordinates second joint (end of link1)
        x1 = origin[0] + int(self.link_length * np.cos(theta1) * scale)
        y1 = origin[1] + int(self.link_length * np.sin(theta1) * scale)
        joint = (x1, y1)
        # Coordinates end-effector
        x2 = x1 + int(self.link_length * np.cos(theta1 + theta2) * scale)
        y2 = y1 + int(self.link_length * np.sin(theta1 + theta2) * scale)
        end_effector = (x2, y2)

        # Draw links
        pygame.draw.line(self.screen, (0, 0, 0), origin, joint, 4)
        pygame.draw.line(self.screen, (0, 0, 0), joint, end_effector, 4)
        # Draw joints and end-effector
        pygame.draw.circle(self.screen, (255, 0, 0), origin, 5)
        pygame.draw.circle(self.screen, (0, 255, 0), joint, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), end_effector, 5)

        pygame.display.flip()

    def close(self) -> None:
        """
        Closes the rendering window and releases pygame resources.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    @abstractmethod
    def params_to_dict(self) -> Dict[str, Any]:
        """
        Converts hyperparameters into a dictionary.
        """
        pass


class TrueReacherEnv(ReacherEnv):
    """
    The "true" reacher environment, simulating the real system dynamics.

    This environment models the physical reacher with exact parameters, 
    serving as the ground truth for comparison with the learned environment.
    """
    def __init__(
        self, link_length: float = 0.5, time_step: float = 0.1, noise_var: float = 0.0
    ) -> None:
        """
        Initialize "true" reacher environment.

        Args:
            time_step (float): Discretization time step.
            noise_var (float): Variance of Gaussian noise added to the state.
        """
        super().__init__(link_length=link_length)
        self.time_step = time_step
        self.noise_var = noise_var

        # Keep track of angular velocities [dtheta1, dtheta2] to simplify calculations
        # of dynamics
        self.dtheta = np.zeros(2, dtype=np.float32)

    def step(
        self, action: np.array
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Parse the old angles and angular velocities
        cos_t1, cos_t2, sin_t1, sin_t2, _, _ = self.state
        theta1_old = np.arctan2(sin_t1, cos_t1)
        theta2_old = np.arctan2(sin_t2, cos_t2)
        dtheta1_old, dtheta2_old = self.dtheta

        # Simple torque integration: 
        #    dtheta_i_new = dtheta_i_old + dt * tau_i
        #    theta_i_new  = theta_i_old  + dt * dtheta_i_new
        tau1, tau2 = np.clip(action, self.action_space.low, self.action_space.high)
        dtheta1_new = dtheta1_old + self.time_step * tau1
        dtheta2_new = dtheta2_old + self.time_step * tau2
        self.dtheta = np.array([dtheta1_new, dtheta2_new], dtype=np.float32)
        theta1_new = theta1_old + self.time_step * dtheta1_new
        theta2_new = theta2_old + self.time_step * dtheta2_new

        # Recompute cos/sin
        cos_t1_new = np.cos(theta1_new)
        sin_t1_new = np.sin(theta1_new)
        cos_t2_new = np.cos(theta2_new)
        sin_t2_new = np.sin(theta2_new)

        # Recompute new (vx, vy) from forward kinematics: 
        #    [vx, vy]^T = J(theta_new) * [dtheta1_new, dtheta2_new]^T
        J_new = self.compute_jacobian(theta1_new, theta2_new)
        v_new = J_new.dot(np.array([dtheta1_new, dtheta2_new], dtype=np.float32))
        vx_new, vy_new = v_new

        # Update the state
        self.state = np.array([
            cos_t1_new, cos_t2_new,
            sin_t1_new, sin_t2_new,
            vx_new, vy_new
        ], dtype=np.float32)

        # Reward is not needed for the true system
        reward = 0.0

        # No termination and truncation conditions defined
        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def compute_jacobian(self, theta1: float, theta2: float) -> np.ndarray:
        """
        Computes the Jacobian matrix of the end-effector position 
          x = l1*cos(t1) + l2*cos(t1 + t2)
          y = l1*sin(t1) + l2*sin(t1 + t2)
        with respect to the joint angles.

        Args:
            theta1 (float): Angle of the first joint (in radians).
            theta2 (float): Angle of the second joint (in radians).

        Returns:
            np.ndarray: A 2x2 Jacobian matrix where:
                J = [[dx/dt1, dx/dt2],
                    [dy/dt1, dy/dt2]]
        """
        dxdt1 = -self.link_length * np.sin(theta1) - self.link_length * np.sin(theta1 + theta2)
        dxdt2 = -self.link_length * np.sin(theta1 + theta2)
        dydt1 =  self.link_length * np.cos(theta1) + self.link_length * np.cos(theta1 + theta2)
        dydt2 =  self.link_length * np.cos(theta1 + theta2)
        return np.array([[dxdt1, dxdt2],
                         [dydt1, dydt2]], dtype=np.float32)

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
        self.state = np.zeros(6, dtype=np.float32)
        self.dtheta = np.zeros(2, dtype=np.float32)
        return self.state, {}
    
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

    def get_state_bounds(
        self, horizon: int, bound_shrink_factor: float
    ) -> Dict[int, np.array]:
        """
        Computes and retrieves state bounds over the specified horizon.

        Args:
            horizon (int): Number of steps to simulate for each action.
            bound_shrink_factor (float): Factor by which to shrink maximum/minimum velocity.

        Returns:
            Dict[int, np.array]: Dictionary mapping state dimension index to their
                                 sampling bounds.
        """
        adjusted_state_bounds = {}

        # Set fixed bounds for angle dimensions
        for dim_idx in [0, 1, 2, 3]:
            adjusted_state_bounds[dim_idx] = np.array([-1.0, 1.0], dtype=np.float32)

        # Compute and shrink raw minimum/maximum state bounds for velocity dimension 
        state_bounds = compute_state_bounds(env=self, horizon=horizon)
        for dim_idx in [4, 5]:
            adjusted_state_bounds[dim_idx] = bound_shrink_factor * state_bounds[dim_idx]

        return adjusted_state_bounds
    
    def sample_states(
        self, num_samples: int, sampling_bounds: Dict[int, np.array]
    ) -> np.array:
        """
        Samples a specified number of states.

        Args:
            num_samples (int): Number of states to sample.
            sampling_bounds (Dict[int, np.array]): Bounds for each state dimension, 
                where each entry is [min, max].

        Returns:
            np.array: An array of shape (num_samples, state_dim) containing the sampled states.
        """
        # Initialize the output array
        sampled_states = np.zeros((num_samples, self.state_dim), dtype=np.float32)

        # Uniformly sample angles theta1 and theta2 from [-π, π]
        theta1_samples = np.random.uniform(-np.pi, np.pi, size=num_samples)
        theta2_samples = np.random.uniform(-np.pi, np.pi, size=num_samples)

        # Convert angles to cosine/sine
        cos_t1 = np.cos(theta1_samples)
        sin_t1 = np.sin(theta1_samples)
        cos_t2 = np.cos(theta2_samples)
        sin_t2 = np.sin(theta2_samples)

        # Uniformely sample velocities within specified bounds
        vx_low, vx_high = sampling_bounds[4]
        vy_low, vy_high = sampling_bounds[5]
        velocity_x_samples = np.random.uniform(vx_low, vx_high, size=num_samples)
        velocity_y_samples = np.random.uniform(vy_low, vy_high, size=num_samples)

        # Combine everything into the sampled states array
        sampled_states[:, 0] = cos_t1
        sampled_states[:, 1] = cos_t2
        sampled_states[:, 2] = sin_t1
        sampled_states[:, 3] = sin_t2
        sampled_states[:, 4] = velocity_x_samples
        sampled_states[:, 5] = velocity_y_samples

        return sampled_states

    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "link_length": self.link_length,
            "time_step": self.time_step
        }
        return parameter_dict


class LearnedReacherEnv(ReacherEnv, LearnedEnv):
    """
    A reacher environment with a learned dynamics model.
    
    This environment simulates the reacher's dynamics using a learned model
    (e.g., Bayesian Neural Network or Feedforward Neural Network) to predict
    the next state and reward.
    """
    def __init__(self, model: Union[FeedforwardNN, BNN]) -> None:
        """
        Initialize "learned" reacher environment.
        """
        LearnedEnv.__init__(self, model=model)
        ReacherEnv.__init__(self)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return LearnedEnv.step(self, action)

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
        self.state = np.zeros(6, dtype=np.float32)
        return self.state, {}

    def params_to_dict(self) -> Dict[str, str]:
        return LearnedEnv.params_to_dict(self)
