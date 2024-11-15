import numpy as np
def random_exploration(env, num_trajectories, horizon):
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_trajectories):
        state, _ = env.reset()
        for _ in range(horizon):    
            states.append(env.unwrapped.state)
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)  
            action = np.expand_dims(action, axis=0) if action.ndim < 1 else action
            actions.append(action)
            next_states.append(next_state)
            
            if terminated or truncated:
                break
    
    return np.array(states), np.array(actions), np.array(next_states)