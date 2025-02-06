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
    Abstract base class for a 2-joint Reacher environment whose state is:
        [cos(theta1), cos(theta2), sin(theta1), sin(theta2), dtheta1, dtheta2].
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
        # [cos(theta1), cos(theta2), sin(theta1), sin(theta2), dtheta1, dtheta2]
        self.state = np.zeros(6, dtype=np.float32)
        self.state_dim = 6

        self.state_dim_names = {
            0: "cos(theta1)",
            1: "cos(theta2)",
            2: "sin(theta1)",
            3: "sin(theta2)",
            4: "dtheta1",
            5: "dtheta2",
        }

        # Two torques for both joints
        torque_limit = 0.3
        self.action_space = spaces.Box(
            low=-torque_limit,
            high=torque_limit,
            shape=(2,),
            dtype=np.float32
        )
        self.action_dim = 2

        # cos/sin are in [-1,1], angular velocities are unbounded
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf], dtype=np.float32),
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
    def set_state(self, state: np.ndarray) -> None:
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
    The "true" reacher environment, simulating the real system dynamics
    with angular velocities in the state.
    """

    def __init__(
        self, link_length: float = 0.5, time_step: float = 0.15, noise_var: float = 0.0
    ) -> None:
        """
        Initialize the true Reacher environment.

        Args:
            link_length (float): Length of each link.
            time_step (float): Discretization time step.
            noise_var (float): Variance of Gaussian noise added to the state.
        """
        super().__init__(link_length=link_length)
        self.time_step = time_step
        self.noise_var = noise_var

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Evolve the system by one time step using the applied torques (action).
        The state is [cos(t1), cos(t2), sin(t1), sin(t2), dtheta1, dtheta2].
        """
        # Extract angles and angular velocities from the current state
        cos_t1, cos_t2, sin_t1, sin_t2, dtheta1_old, dtheta2_old = self.state

        theta1_old = np.arctan2(sin_t1, cos_t1)
        theta2_old = np.arctan2(sin_t2, cos_t2)

        # Clip the action to valid torque range
        tau1, tau2 = np.clip(action, self.action_space.low, self.action_space.high)

        # Integrate angular velocities:
        dtheta1_new = dtheta1_old + self.time_step * tau1
        dtheta2_new = dtheta2_old + self.time_step * tau2

        # Integrate angles:
        theta1_new = theta1_old + self.time_step * dtheta1_new
        theta2_new = theta2_old + self.time_step * dtheta2_new

        # Recompute cos/sin
        cos_t1_new = np.cos(theta1_new)
        sin_t1_new = np.sin(theta1_new)
        cos_t2_new = np.cos(theta2_new)
        sin_t2_new = np.sin(theta2_new)

        # Optionally add noise
        if self.noise_var > 0.0:
            dtheta1_new += np.random.normal(0, np.sqrt(self.noise_var))
            dtheta2_new += np.random.normal(0, np.sqrt(self.noise_var))

        # Update the environment state
        self.state = np.array([
            cos_t1_new,
            cos_t2_new,
            sin_t1_new,
            sin_t2_new,
            dtheta1_new,
            dtheta2_new,
        ], dtype=np.float32)

        # Reward is not needed for the true system
        reward = 0.0

        # No termination and truncation conditions defined
        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def set_state(self, state: np.ndarray) -> None:
        """
        Sets the environment's state directly. The expected state is:
          [cos(theta1), cos(theta2), sin(theta1), sin(theta2), dtheta1, dtheta2].
        """
        self.state = state.astype(np.float32)

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to an initial configuration.

        Returns:
            Tuple[np.ndarray, dict]: Initial state and additional reset info.
        """
        super().reset(seed=seed)
        # For example, start at theta1=0, theta2=0, with zero velocities
        self.state = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.state, {}

    def get_action_bounds(self) -> Dict[int, np.ndarray]:
        """
        Retrieves the action bounds for each action dimension.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping action dimension index to [min, max].
        """
        action_bounds = {}
        for dim_idx in range(self.action_dim):
            action_bounds[dim_idx] = np.array(
                [self.action_space.low[dim_idx], self.action_space.high[dim_idx]],
                dtype=np.float32
            )
        return action_bounds

    def get_state_bounds(
        self, horizon: int, bound_shrink_factor: float
    ) -> Dict[int, np.ndarray]:
        """
        Computes and retrieves state bounds over the specified horizon.
        For cos/sin, we fix [-1,1]. For angular velocities, we can compute
        some plausible bounds via a short simulation, or set manually.
        """
        adjusted_state_bounds = {}

        # Angles (cos/sin) are always in [-1, 1]
        for dim_idx in [0, 1, 2, 3]:
            adjusted_state_bounds[dim_idx] = np.array([-1.0, 1.0], dtype=np.float32)

        # If desired, you can compute actual dtheta bounds with a small rollout
        # or just place a large range. Below is an example approach:
        # We'll use compute_state_bounds, but remember it won't directly apply to cos/sin-based states.
        state_bounds = compute_state_bounds(env=self, horizon=horizon)
        for dim_idx in [4, 5]:  # dtheta1, dtheta2
            adjusted_state_bounds[dim_idx] = bound_shrink_factor * state_bounds[dim_idx]

        return adjusted_state_bounds

    def sample_states(
        self, num_samples: int, sampling_bounds: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Samples a specified number of states. In this version, we sample random angles
        and random angular velocities within the provided bounds.
        """
        sampled_states = np.zeros((num_samples, self.state_dim), dtype=np.float32)

        # Sample angles theta1, theta2 uniformly from [-π, π]
        theta1_samples = np.random.uniform(-np.pi, np.pi, size=num_samples)
        theta2_samples = np.random.uniform(-np.pi, np.pi, size=num_samples)

        # Convert angles to cosine/sine
        cos_t1 = np.cos(theta1_samples)
        sin_t1 = np.sin(theta1_samples)
        cos_t2 = np.cos(theta2_samples)
        sin_t2 = np.sin(theta2_samples)

        # Sample dtheta1, dtheta2 from the sampling bounds
        dtheta1_low, dtheta1_high = sampling_bounds[4]
        dtheta2_low, dtheta2_high = sampling_bounds[5]
        dtheta1 = np.random.uniform(dtheta1_low, dtheta1_high, size=num_samples)
        dtheta2 = np.random.uniform(dtheta2_low, dtheta2_high, size=num_samples)

        # Fill in
        sampled_states[:, 0] = cos_t1
        sampled_states[:, 1] = cos_t2
        sampled_states[:, 2] = sin_t1
        sampled_states[:, 3] = sin_t2
        sampled_states[:, 4] = dtheta1
        sampled_states[:, 5] = dtheta2

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
    A reacher environment with a learned dynamics model. The state is
    [cos(theta1), cos(theta2), sin(theta1), sin(theta2), dtheta1, dtheta2].
    """

    def __init__(self, model: Union[FeedforwardNN, BNN]) -> None:
        """
        Initialize the learned Reacher environment.
        """
        LearnedEnv.__init__(self, model=model)
        ReacherEnv.__init__(self)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Defer to LearnedEnv.step(), which will use the model to predict the next state.
        return LearnedEnv.step(self, action)
    
    def set_state(self, state: np.ndarray) -> None:
        # Defer to LearnedEnv.set_state()
        return LearnedEnv.set_state(self, state)

    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the learned environment to a default state (theta=0, dtheta=0).
        """
        super().reset(seed=seed)
        self.state = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.state, {}

    def params_to_dict(self) -> Dict[str, Any]:
        return LearnedEnv.params_to_dict(self)
