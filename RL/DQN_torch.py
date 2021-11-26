# -*- codeing = utf-8 -*-
# @Time : 2021/11/21 11:32
# @Author : Evan_wyl
# @File : DQN_torch.py

import gym
from gym import wrappers

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta
import math
import os
import pickle


config = dict()
config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["epsilon_start"] = 1.0
config["epsilon_final"] = 0.01
config["epsilon_decay"] = 30000
config["epsilon_by_frame"] = lambda frame_idx: config["epsilon_start"] - (config["epsilon_start"] - config["epsilon_final"]) \
                                            * math.exp(-1 * config["epsilon_decay"] / frame_idx)

config["GAMMA"] = 0.99
config["LR"] = 1e-4

config["TARGET_NET_UPDATE_FREQ"] = 1000
config["EXP_REPLAY_SIZE"] = 100000
config["BATCH_SIZE"] = 32

config["LEARN_START"] = 10000
config["MAX_FRAMES"] = 1000000
config["TARGET_NET_UPDATE_FREQ"] = 1000
config["EXP_REPLAY_SIZE"] = 100000
config["BATCH_SIZE"] = 32
config["LEARN_START"] = 10000


class ExperienceReplay_Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memroy = []

    def push(self, transition):
        self.memroy.append(transition)
        if len(self.memroy) > self.capacity:
            del  self.memroy[0]

    def sample(self, batch_size):
        return random.sample(self.memroy, batch_size)

    def __len__(self):
        return len(self.memroy)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

class BaseAgent(object):
    def __init__(self):
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.losses = []
        self.rewards = []
        self.sigma_parameter_mag = []

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def save_w(self):
        torch.save(self.model.state_dict(), './saved_agents/model.dump')
        torch.save(self.optimizer.state_dict(), "./saved_agents/optim.dump")

    def load_w(self):
        fname_model = "./saved_agents/model.dump"
        fname_optim = "./saved_agents/optim.dump"

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict)
        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))

    def save_replay(self):
        pickle.dump(self.memory, open("./saved_agents/exp_replay_agent.dump", "wb"))

    def load_replay(self):
        fname = "./saved_agents/exp_replay_agent.dump"
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, "rb"))

    def save_sigma_param_magnitudes(self):
        tmp = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "sigma" in name:
                    tmp += param.data.cpu().numpy().ravel().tolist()
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    def save_loss(self, loss):
        self.losses.append(loss)

    def save_reward(self, reward):
        self.rewards.append(reward)

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__()
        self.device = config["device"]

        self.gamma = config['GAMMA']
        self.lr = config["LR"]
        self.target_net_update_freq = config["TARGET_NET_UPDATE_FREQ"]
        self.experience_replay_size = config["EXP_REPLAY_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.learn_start = config["LEARN_START"]

        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0
        self.declare_memory()

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplay_Memory(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batcg_next_state = zip(*transitions)
        shape = (-1, ) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_state, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batcg_next_state)), device=self.device, dtype=torch.uint8)
        try:
            non_final_next_states = torch.tensor([s for s in batcg_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        return batch_state, batch_action, batch_reward, non_final_mask, non_final_next_states, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_mask, non_final_next_states, empty_next_state_values = batch_vars

        current_q_values = self.model(batch_state).gather(1, batch_action)

        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + (self.gamma*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()
        return loss

    def update(self, s, a , r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)
        if frame < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()
        loss = self.compute_loss(batch_vars=batch_vars)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, non_final_next_states):
        return self.target_model(non_final_next_states).max(dim=1)[1].view(1,1)

    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)



