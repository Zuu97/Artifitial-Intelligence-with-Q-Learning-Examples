environment = 'FrozenLake8x8-v0'
num_episodes = 50000
max_steps_per_episode = 100
test_episodes = 100

learning_rate = 0.1
discount_rate = 0.99
verbose = 1000
eps = 1
max_eps = 1
min_eps = 0.01
eps_decay = 0.001

q_table_path = "FrozenLake8x8-v0.npy"

bonus = 600
panelty = -400
winning_state = 63