import numpy as np
from variables import*

def create_state_bin(env,state):

    car_position_bin = np.linspace(position_min,position_max, num=bin_count)
    car_velocity_bin = np.linspace(velocity_min,velocity_max, num=bin_count)

    bins = [car_position_bin, car_velocity_bin]
    return np.array([np.digitize(s, bins[i]) for i,s in enumerate(state)])

def new_state_space(env,state):
    discrete_state = create_state_bin(env,state)
    state2array = discrete_state.tolist()
    return int(''.join(list(map(str,state2array))))

def new_action_space(env,action):

    action_bin = np.linspace(action_min,action_max, num=bin_count)
    return np.digitize(action[0], action_bin)


def discrete2cts(action):
    bin_size = (action_max - action_min)/(bin_count)
    marginal_value = (bin_size * action) - 0.95
    return [marginal_value]
