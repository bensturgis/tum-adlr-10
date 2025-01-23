import sys
import os
sys.path.append(os.path.abspath("."))

import numpy as np
import pygame
import matplotlib.pyplot as plt

from dynamical_systems.reacher import TrueReacherEnv

true_reacher_env = TrueReacherEnv()

state = true_reacher_env.reset()
state_history = []
clock = pygame.time.Clock()
for _ in range(500):
    action = np.array([0.1, 0.1])
    new_state, _, _, _, _= true_reacher_env.step(action)
    state_history.append(new_state)
    true_reacher_env.render()
    clock.tick(100)

true_reacher_env.close()
plt.plot(state_history)
plt.show()