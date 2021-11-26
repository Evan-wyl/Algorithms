# -*- codeing = utf-8 -*-
# @Time : 2021/11/26 15:15
# @Author : Evan_wyl
# @File : reinforce.py

import argparse
import gym
import numpy as np
from itertools import count

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.distributions import Categorical

config = dict()
config['gamma'] = 0.99
config['seed'] = 543
config['render'] = False
config['log-intervel'] = 10

env = gym.make('CartPole-v1')
env.seed(config['seed'])
torch.manual_seed(config['seed'])

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(4,128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128,2)

        self.saved_log_prods = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        scores = self.fc2(x)
        return F.softmax(scores, dim=1)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
# eps为无穷小
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_prods.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + config["gamma"] * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # REINFORCE主要代码
    for log_prob, R in zip(policy.saved_log_prods, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_prods[:]

if __name__ == '__main__':
    for i_ep in range(10000):
        state, ep_reward = env.reset(), 0
        while True:
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if config["render"]:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        finish_episode()

