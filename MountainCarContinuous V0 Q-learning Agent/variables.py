environment = 'MountainCarContinuous-v0'
bin_count = 9

goal_positon = 0.5

position_min = -1.2
position_max = 0.6

velocity_min = -0.07
velocity_max = 0.07

action_min = -0.95
action_max = 1.0

num_episodes = 10000
max_steps_in_episode = 200
verbose = 100
learning_rate = 10e-3
eps = 1
discount_factor = 0.9
panelty = -1000
test_episodes = 100
bonus = 200

q_table_path="MountainCarContinuous-v0.npy"