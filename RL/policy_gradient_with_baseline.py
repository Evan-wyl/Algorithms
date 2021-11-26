# -*- codeing = utf-8 -*-
# @Time : 2021/11/26 15:16
# @Author : Evan_wyl
# @File : policy_gradient_with_baseline.py

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
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(200, 1)

        self.saved_log_prods = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.action_head(x)
        states_score = self.value_head(x)
        return F.softmax(action_scores, dim=1), states_score

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
# eps为无穷小
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, states_values = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_prods.append((m.log_prob(action), states_values))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    value_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + config["gamma"] * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    # REINFORCE主要代码
    for (log_prob,states_val), reward in zip(policy.saved_log_prods, returns):
        policy_loss.append(-log_prob * (reward - states_val))
        # L1 loss和L2 loss的结合
        value_loss.append(F.smooth_l1_loss(states_val, reward))

    optimizer.zero_grad()
    # cat和stack在这里的作用一致, 都把把policy_loss和value_loss转换成张量
    # cat:把两个concate在一起
    # stack:把两个张量通过增加一个维度的方式合并在一起
    policy_loss = torch.cat(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss
    loss.backward()
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

