from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np

def get_action_obs_structure(env):

    # returns an action list, action_is_scalar for action, and number of obs and obs_is_scalar

    if isinstance(env.action_space, Discrete):
        action, action_is_scalar = [env.action_space.n], True
    elif isinstance(env.action_space, MultiDiscrete):
        action, action_is_scalar = env.action_space.nvec, False
    elif isinstance(env.action_space, Box): 
        # in absence of better options we are just taking the high values
        action, action_is_scalar = env.action_space.high, False 
    else:
        raise NotImplementedError(f"Unsupported space type: {type(env.action_space)}")

    if isinstance(env.observation_space, Discrete):
        n_obs, obs_is_scalar = 1, True
    elif isinstance(env.observation_space, MultiDiscrete):
        n_obs, obs_is_scalar = len(env.observation_space.nvec), False
    elif isinstance(env.observation_space, Box):
        # in absence of better options we are just taking he high values
        n_obs, obs_is_scalar = np.prod(env.observation_space.shape), False
    else:
        raise NotImplementedError(f"Unsupported space type: {type(env.observation_space)}")
    
    return action, action_is_scalar, n_obs, obs_is_scalar

def get_action_obs_structure_for_discrete(env):

    # returns an action list, action_is_scalar for action, and obs list and obs_is_scalar

    if isinstance(env.action_space, Discrete):
        action, action_is_scalar = [env.action_space.n], True
    elif isinstance(env.action_space, MultiDiscrete):
        action, action_is_scalar = env.action_space.nvec, False
    else:
        raise NotImplementedError(f"Unsupported space type for discrete agent: {type(env.action_space)}")

    if isinstance(env.observation_space, Discrete):
        obs, obs_is_scalar = [env.observation_space.n], True
    elif isinstance(env.observation_space, MultiDiscrete):
        obs, obs_is_scalar = env.observation_space.nvec, False
    else:
        raise NotImplementedError(f"Unsupported space type for discrete agent: {type(env.observation_space)}")
    
    return action, action_is_scalar, obs, obs_is_scalar