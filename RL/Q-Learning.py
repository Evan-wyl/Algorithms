# -*- codeing = utf-8 -*-
# @Time : 2021/11/11 10:46
# @Author : Evan_wyl
# @File : Q-Learning.py

import gym
import numpy as np
import time
from gridworld import CliffWalkingWapper


class QLeaningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))

    def sample(self, obs):
        if np.random.uniform(0, 1) < (1 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)


    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')


    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    obs = env.reset()

    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward
        total_steps += 1
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs

        if done:
            break
    return total_reward

if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    env = CliffWalkingWapper(env)
    agent = QLeaningAgent(
        obs_n = env.observation_space.n,
        act_n = env.action_space.n,
        learning_rate = 0.1,
        gamma = 0.9,
        e_greed = 0.1
    )

    for episode in range(50000):
        ep_reward, ep_steps = run_episode(env, agent, True)
        print('Episode %s: step = %s, reward = %.1f' % (episode, ep_steps, ep_reward))

    test_reward = test_episode(env, agent)
    print('test_reward = %.1f' % (test_reward))



