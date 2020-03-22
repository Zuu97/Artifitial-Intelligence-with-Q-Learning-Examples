import os
import gym
import numpy as np
from matplotlib import pyplot as plt
import random
from variables import *
from util import new_state_space
from IPython import display

class CartPoleV0Agent(object):
    def __init__(self):
        self.env = gym.make(environment)

        low_state_vale = new_state_space(self.env, self.env.observation_space.low)
        high_state_vale = new_state_space(self.env, self.env.observation_space.high)

        num_states = high_state_vale - low_state_vale + 1
        num_actions = self.env.action_space.n

        self.q_table = np.random.uniform(low = -1, high = 1 , size = (num_states,num_actions))

    def train(self):
        global eps
        total_rewards_in_episodes = []
        for episode in range(num_episodes):
            done = False
            time_steps = 0
            episode_reward = 0
            state = new_state_space(self.env,self.env.reset())
            while not done :
                if random.uniform(0,1) < eps:
                    action =  self.env.action_space.sample()
                else:
                    action =  np.argmax(self.q_table[state,:])

                new_state, reward, done, _ = self.env.step(action)

                new_state = new_state_space(self.env,new_state)

                time_steps += 1

                if time_steps < panelty_time_steps and  done:
                    reward = panelty # add panelty if terminate before ending the episode

                self.q_table[state,action] = (1 - learning_rate) * self.q_table[state,action] \
                                                + learning_rate * (reward + discount_factor * np.max(self.q_table[new_state,:]))

                state = new_state
                episode_reward += reward

            if (episode + 1)% verbose == 0:
                print("episode :", episode+1," reward: ",episode_reward)

            eps = eps / np.sqrt(episode+1)

            total_rewards_in_episodes.append(episode_reward)

        CartPoleV0Agent.plot_cumulative_rewards(total_rewards_in_episodes,num_episodes)

    @staticmethod
    def plot_cumulative_rewards(total_rewards_in_episodes,num_episodes):
        # plot the cumulative average rewards
        cum_rewards = np.cumsum(total_rewards_in_episodes)
        cum_average_reward = cum_rewards / np.arange(1,num_episodes+1)

        fig = plt.figure()
        plt.plot(cum_average_reward)
        fig.suptitle('Q learning reward analysis', fontsize=20)
        plt.xlabel('episode number')
        plt.ylabel('Cumulative Average Reward')
        fig.savefig('frozen_lake.png')

    def test(self):
        win = 0
        loss = 0
        for episode in range(test_episodes):
            state = self.env.reset()
            print("episode :", episode+1)
            time_steps = 0
            for step in range(max_steps_in_episode):
                self.env.render()
                state = new_state_space(self.env,state)
                action = np.argmax(self.q_table[state,:])
                new_state, reward, done, info = self.env.step(action)

                if done:
                    self.env.render()
                    if time_steps < panelty_time_steps:
                        loss += 1
                        print("YOU LOOSER !!!")
                    else:
                        win += 1
                        print("YOU WIN !!!")
                    break
                time_steps += 1
                state = new_state
        print("Winning Percentage : ",win/(win + loss))
        self.env.close()

    def save_q_table(self):
        np.save(q_table_path, self.q_table)

    def load_q_table(self):
        self.q_table = np.load(q_table_path)

if __name__ == "__main__":
    agent =  CartPoleV0Agent()
    if os.path.exists(q_table_path):
        agent.load_q_table()
    else:
        agent.train()
        agent.save_q_table()
    agent.test()
