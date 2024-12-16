import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import platform
import ctypes
import torch
from typing import Dict, Tuple, Any

from models.feedforward_nn import FeedforwardNN
from models.mc_dropout_bnn import MCDropoutBNN

class MassSpringDamperEnv(gym.Env):
    def __init__(
            self, m=0.1, k=1.0, d=0.1, delta_t=0.01, nlin=False, noise_var=0.1, model=None
    ):
        super(MassSpringDamperEnv, self).__init__()

        # Physical parameters
        self.nonlinear = nlin
        self.noise_var = noise_var
        self.model = model
        self.m = m
        self.k = k
        self.d = d
        self.delta_t = delta_t  # Time step for discretization
        self.input_limit = 1.0
        self.name = "Mass-Spring-Damper System"

        # State: [position, velocity]
        self.state = np.array([0.0, 0.0])

        # Action space: Force input with constraints
        self.action_space = spaces.Box(low=-self.input_limit, high=self.input_limit, shape=(1,), dtype=np.float32)

        # Observation space: [position, velocity]
        self.observation_space = spaces.Box(low=-6, high=6, shape=(2,), dtype=np.float32)

        # Initialize pygame for rendering
        self.screen = None
        self.width, self.height = 600, 200
        self.mass_pos = self.width // 2  # Initial mass position in the middle
        self.spring_origin = self.mass_pos - 200  # Fixed point for spring start
    
    def reset(self, state=None):
        # Reset state to given conditions
        if state is not None:
            self.state = state
            return self.state, {}
        # Reset state to initial conditions
        self.state = np.array([0.0, 0.0])  # Start at rest
        return self.state, {}

    def step(self, action: np.array) -> Tuple[Any, float, bool, bool, dict]:
        # Unpack state
        x, v = self.state

        # Apply force (action)
        F = action.squeeze()

        # Apply model when exists
        if self.model is not None:
            if isinstance(self.model, MCDropoutBNN):
                s = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)  # torch.Size([1, 2])
                a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # torch.Size([1, 1])
                next_state, pred_var = self.model.bayesian_pred(s, a)
                self.state = next_state.squeeze()
                
                reward = -np.sum(np.square(self.state))
                terminated = False
                truncated = False
                info = {"var": pred_var.squeeze()} # output variance of predictive distribution

                return self.state, reward, terminated, truncated, info
            
            if isinstance(self.model, FeedforwardNN):
                s = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
                a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                next_state = self.model(s, a)
                self.state = next_state.squeeze(0).detach().numpy()
                
                reward = -np.sum(np.square(self.state))
                terminated = False
                truncated = False

                return self.state, reward, terminated, truncated, {}

        # Non-linear stiffness
        if self.nonlinear:
            f_k = np.tanh(x)
            x_ = x + self.delta_t * v
            v_ = v + self.delta_t * (-self.d * v - f_k + F) / self.m
            self.state = np.array([x_, v_])
        else:
            A = np.array([[0, 1], [-self.k / self.m, -self.d / self.m]])
            B = np.array([0, 1 / self.m])

            self.state = (np.eye(2) + self.delta_t * A) @ self.state + self.delta_t * B * F
        self.state = np.random.normal(loc=self.state, scale=self.noise_var, size=(2,)) # add gaussian noise

        # Calculate reward (deviation from origin)
        reward = -np.sum(np.square(self.state))

        terminated = False # no terminal state defined
        truncated = False # maybe when x exceeds some bounds

        return self.state, reward, terminated, truncated, {}

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
            self.reset(state=np.array([0.0, 0.0]))
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
            "min_velocity": overall_min_vel
        }

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Mass-Spring-Damper System")
            # Set window to be always on top
            if platform.system() == "Windows":
                hwnd = pygame.display.get_wm_info()['window']
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001)

        self.screen.fill((255, 255, 255))

        # Parameters for visualization
        x = self.state[0]
        mass_x = int(self.mass_pos + x * 25)  # Scale position for visualization

        pygame.draw.line(self.screen, (0, 0, 0), (self.spring_origin, self.height // 2 - 20), (self.spring_origin, self.height // 2 + 20), 4)

        # Draw spring as a line
        pygame.draw.line(self.screen, (0, 0, 0), (self.spring_origin, self.height // 2), (mass_x, self.height // 2), 2)

        # Draw mass as a rectangle
        mass_rect = pygame.Rect(mass_x - 10, self.height // 2 - 10, 20, 20)
        pygame.draw.rect(self.screen, (0, 255, 0), mass_rect)

        pygame.display.flip()

    def params_to_dict(self) -> Dict[str, str]:
        """
        Converts hyperparameters into a dictionary.
        """
        parameter_dict = {
            "name": self.name,
            "m": self.m,
            "k": self.k,
            "d": self.d,
            "delta_t": self.delta_t,
            "nonlinear": self.nonlinear,
            "noise_var": self.noise_var,
            "model": None if self.model is None else self.model.params_to_dict()
        }
        return parameter_dict

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
