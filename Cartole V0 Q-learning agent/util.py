import numpy as np
from variables import *

def create_bin(env,state):

    cart_position_bin = np.linspace(cart_position_min,cart_position_high, num=bin_count)
    car_velocity_bin = np.linspace(cart_velocity_min,cart_velocity_high, num=bin_count)
    pole_angle_bin = np.linspace(pole_angle_min,pole_angle_high, num=bin_count)
    pole_velocity_bin = np.linspace(pole_velocity_min,pole_velocity_high, num=bin_count)

    bins = [cart_position_bin, car_velocity_bin, pole_angle_bin, pole_velocity_bin]
    return np.array([np.digitize(s, bins[i]) for i,s in enumerate(state)])

def new_state_space(env,state):
    discrete_state = create_bin(env,state)
    state2array = discrete_state.tolist()
    return int(''.join(list(map(str,state2array))))
