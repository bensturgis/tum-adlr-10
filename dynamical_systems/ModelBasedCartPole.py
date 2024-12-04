import numpy as np
import torch
import matplotlib.pyplot as plt
from gymnasium.envs.classic_control import CartPoleEnv

class ModelBasedCartPoleEnv(CartPoleEnv):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def step(self, action):
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predicted_next_state = self.model(state_tensor, action_tensor)

        self.state = predicted_next_state.squeeze(0).numpy()

        # set bound as original CartPoleEnv
        # self.state[0] = np.clip(self.state[0], -4.8, 4.8)
        # self.state[1] = np.clip(self.state[1], -np.inf, np.inf)
        # self.state[2] = np.clip(self.state[2], -0.418, 0.418)
        # self.state[3] = np.clip(self.state[3], -np.inf, np.inf)

        reward = 1.0
        terminated = bool(
            self.state[0] < -4.8
            or self.state[0] > 4.8
            or self.state[2] < -0.418
            or self.state[2] > 0.418
        )
        truncated = False

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}
