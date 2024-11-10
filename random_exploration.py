def random_exploration(env, num_trajectories, horizon):
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_trajectories):
        state, _ = env.reset()
        for _ in range(horizon):    
            states.append(env.state)
            action = env.action_space.sample()
            actions.append(action)
            next_state, reward, terminated, truncated, info = env.step(action)  
            next_states.append(next_state)
            
            if terminated or truncated:
                break

    return states, actions, next_states