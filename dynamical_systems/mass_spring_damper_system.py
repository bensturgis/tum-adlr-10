import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import platform
import ctypes
import torch
from typing import Tuple, Any

from models.feedforward_nn import FeedforwardNN
from models.mc_dropout_bnn import MCDropoutBNN

class MassSpringDamperEnv(gym.Env):
    def __init__(self, m=0.1, k=1.0, d=0.1, delta_t=0.01, nlin=False, noise_var=0.1,model=None):
        super(MassSpringDamperEnv, self).__init__()

        # Physical parameters
        self.nonlinear = nlin
        self.noise_var = noise_var
        self.model = model
        self.m = m
        self.k = k
        self.d = d
        self.delta_t = delta_t  # Time step for discretization
        self.input_limit = 10.0

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
                s = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
                a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                next_state, pred_var = self.model.bayesian_pred(s, a)
                self.state = next_state
                
                reward = -np.sum(np.square(self.state))
                terminated = False
                truncated = False
                info = {"var": pred_var} # output variance of predictive distribution

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

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
