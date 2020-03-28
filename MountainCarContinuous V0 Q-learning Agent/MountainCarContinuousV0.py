import os
import gym
import numpy as np
from matplotlib import pyplot as plt
import random
from variables import *
from util import new_state_space, new_action_space, discrete2cts

class MountainCarContinuousV0(object):
    def __init__(self):
        self.env = gym.make(environment)

        low_state_value = new_state_space(self.env, self.env.observation_space.low)
        high_state_value = new_state_space(self.env, self.env.observation_space.high)

        low_action_value = new_action_space(self.env, self.env.action_space.low)
        high_action_value = new_action_space(self.env, self.env.action_space.high)

        num_states = high_state_value - low_state_value + 1
        num_actions = high_action_value - low_action_value + 1

        self.q_table = np.random.uniform(low = 0, high = 1 , size = (num_states,num_actions))

    def train(self):
        global eps
        total_rewards_in_episodes = []
        for episode in range(num_episodes):
            done = False
            episode_reward = 0
            state = new_state_space(self.env,self.env.reset())
            time_step = 0
            while not done :
                time_step += 1
                if random.uniform(0,1) < eps:
                    action =  self.env.action_space.sample()
                    dis_action = new_action_space(self.env, action)
                else:
                    dis_action =  np.argmax(self.q_table[state,:])
                    action = discrete2cts(dis_action)

                new_state, reward, done, _ = self.env.step(action)

                reward = reward - time_step * 2
                if new_state[0] < goal_positon and  done:
                    reward += panelty # add panelty if terminate before ending the episode

                elif new_state[0] >= goal_positon and  done:
                    reward += bonus

                new_state = new_state_space(self.env,new_state)

                self.q_table[state,dis_action] = (1 - learning_rate) * self.q_table[state,dis_action] \
                                                + learning_rate * (reward + discount_factor * np.max(self.q_table[new_state,:]))

                state = new_state
                episode_reward += reward

            if (episode + 1)% verbose == 0:
                print("episode :", episode+1," reward: ",episode_reward)

            eps = eps / np.sqrt(episode+1)

            total_rewards_in_episodes.append(episode_reward)

        MountainCarContinuousV0.plot_cumulative_rewards(total_rewards_in_episodes,num_episodes)

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
        fig.savefig('MountainCarContinuous-v0.png')

    def test(self):
        win = 0
        loss = 0
        for episode in range(test_episodes):
            state = self.env.reset()
            print("episode :", episode+1)
            for step in range(max_steps_in_episode):
                self.env.render()
                state = new_state_space(self.env,state)
                dis_action = np.argmax(self.q_table[state,:])
                action = discrete2cts(dis_action)
                new_state, reward, done, info = self.env.step(action)

                if done:
                    self.env.render()
                    if new_state[0] >= goal_positon:
                        win += 1
                        print("YOU WIN !!!")
                    else:
                        loss += 1
                        print("YOU LOOSER !!!")
                    break
                state = new_state
        print("Winning Percentage : ",win/(win + loss))
        self.env.close()

    def save_q_table(self):
        np.save(q_table_path, self.q_table)

    def load_q_table(self):
        self.q_table = np.load(q_table_path)

if __name__ == "__main__":
    agent =  MountainCarContinuousV0()
    if os.path.exists(q_table_path):
        agent.load_q_table()
    else:
        agent.train()
        agent.save_q_table()
    agent.test()
