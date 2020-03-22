environment = 'CartPole-v0'
bin_count = 9

cart_position_high  =  2.4
cart_position_min   = -2.4

cart_velocity_high  =  2
cart_velocity_min   = -2

pole_angle_high     =  0.4
pole_angle_min      = -0.4

pole_velocity_high  =  3.5
pole_velocity_min   = -3.5

num_episodes = 10000
max_steps_in_episode = 10000
verbose = 100
learning_rate = 10e-3
eps = 1
discount_factor = 0.9
panelty = -400
panelty_time_steps = 199
test_episodes = 100

q_table_path="cartpole-V0.npy"


