# -*- codeing = utf-8 -*-
# @Time : 2021/12/3 10:01
# @Author : Evan_wyl
# @File : DDPG.py

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            target = self.critic_target(next_state, self.actor_target(next_state))
            target = reward + (self.discount * not_done * target).detach()

        critic_loss = F.mse_loss(target, self.critic(state, action))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        actor_loss = -self.critic(state, self.actor(state))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')

        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + '_critic'))
        self.critic_optimizer.load_state_dict(torch.load(self.critic_optimizer.load_state_dict(filename + '_critic_optimizer')))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.actor_optimizer.load_state_dict(torch.load(filename + '_actor_optimizer'))
        self.actor_target = copy.deepcopy(self.actor)
