import numpy as np
from variables import*

def create_bin(env,state):

    car_position_bin = np.linspace(position_min,position_max, num=bin_count)
    car_velocity_bin = np.linspace(velocity_min,velocity_max, num=bin_count)

    bins = [car_position_bin, car_velocity_bin]
    return np.array([np.digitize(s, bins[i]) for i,s in enumerate(state)])

def new_state_space(env,state):
    discrete_state = create_bin(env,state)
    state2array = discrete_state.tolist()
    return int(''.join(list(map(str,state2array))))
