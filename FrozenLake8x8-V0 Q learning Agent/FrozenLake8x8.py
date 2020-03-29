import os
import gym
gym.logger.set_level(40)
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from variables import*

class FrozenLake8x8(object):
    def __init__(self):
        self.env = gym.make(environment)
        action_space_size = self.env.action_space.n
        state_space_size  = self.env.observation_space.n
        self.q_table = np.random.uniform(low = 0, high = 1 , size = (state_space_size,action_space_size))

    def train(self):
        global eps
        rewards_all_episodes = []
        for episode in range(num_episodes):
            state = self.env.reset()

            done = False
            episode_reward = 0
            step = 0
            while not done:
                step += 1
                r = random.uniform(0,1)
                if r > eps:
                    action = np.argmax(self.q_table[state,:])
                else:
                    action = self.env.action_space.sample()
                new_state, reward, done, info = self.env.step(action)

                if done:
                    if new_state == winning_state:
                        reward += bonus
                    else:
                        reward += panelty
                else:
                    if max_steps_per_episode <= step:
                        reward += panelty
                        done = True

                self.q_table[state,action] = (1 - learning_rate) * self.q_table[state,action] \
                                            + learning_rate*(reward + discount_rate*np.max(self.q_table[new_state,:]))

                state = new_state
                episode_reward += reward


            if (episode + 1)% verbose == 0:
                print("episode :", episode+1," reward: ",episode_reward)

            eps = min_eps + (max_eps - min_eps) * np.exp(-eps_decay * episode)
            rewards_all_episodes.append(episode_reward)

        # plot the cumulative average rewards
        cum_rewards = np.cumsum(rewards_all_episodes)
        cum_average_reward = cum_rewards / np.arange(1,num_episodes+1)

        fig = plt.figure()
        plt.plot(cum_average_reward)
        fig.suptitle('Q learning reward analysis', fontsize=20)
        plt.xlabel('episode number')
        plt.ylabel('Cumulative Average Reward')
        fig.savefig('FrozenLake8x8.png')

    def test(self):
        win = 0
        loss = 0
        for episode in range(test_episodes):
            state = self.env.reset()

            done = False
            print("episode :", episode+1)
            for step in range(max_steps_per_episode):
                self.env.render()

                action = np.argmax(self.q_table[state,:])
                new_state, reward, done, info = self.env.step(action)

                if done:
                    self.env.render()
                    if new_state == winning_state:
                        print("YOU WIN !!!")
                        win += 1
                    else:
                        print("YOU LOOSER !!!")
                        loss += 1
                    break

                state = new_state
            else:
                print("YOU LOOSER !!!")
                loss += 1
        win_acc = win / (win + loss)
        print("Winning Percentage : {}%".format(100*win_acc))
        self.env.close()

    def save_q_table(self):
        np.save(q_table_path, self.q_table)

    def load_q_table(self):
        self.q_table = np.load(q_table_path)
        return np.load(q_table_path)

if __name__ == "__main__":
    agent =  FrozenLake8x8()
    if os.path.exists(q_table_path):
        agent.load_q_table()
    else:
        agent.train()
        agent.save_q_table()
        agent.test()
