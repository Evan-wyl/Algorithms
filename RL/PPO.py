# -*- codeing = utf-8 -*-
# @Time : 2021/12/2 11:12
# @Author : Evan_wyl
# @File : PPO.py

import gym

gym.logger.set_level(40)

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from timeit import default_timer as timer
from datetime import timedelta
import os
import glob

log_dir = '/tmp/gym_ppo/'
try:
    os.mkdir(log_dir)
except OSError:
    # 进行文件路径，并以列表的形式返回
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

config = dict()
config["ppo_epoch"] = 3
config["num_mini_batch"] = 32
config["ppo_clip_param"] = 0.1

config['num_agents'] = 8
config["rollout"] = 128
config['USE_GAE'] = True
config['gae_tau'] = 0.95

config['GAMMA'] = 0.99
config['LR'] = 7e-4
config['emtropy_loss_weight'] = 0.01
config['value_loss_weight'] = 1.0
config['grad_norm_max'] = 0.5
config["device"] = "cuda"

config['MAX_FRAMES'] = int(1e7 / config['num_agents'] / config['rollout'])


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()

        # nn.init.orthogonal_中参数gain：可选比例因子
        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                                          lambda x: nn.init.constant_(x, 0),
                                          nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, kernel_size=3, stride=1))
        self.fc1 = init_(nn.Linear(self.feature_size(input_shape), 512))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                                          lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(512, 1))

        init_ = lambda m: self.layer_init(m,
                                          nn.init.orthogonal_,
                                          lambda x: nn.init.constant_(x, 0),
                                          gain=0.01)

        self.action_linear = init_(nn.Linear(512, num_actions))
        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu((self.fc1(x)))

        value = self.critic_linear(x)
        logits = self.action_linear(x)
        return logits, value

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, device, USE_GAE=True, gae_tau=0.95):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, 1).to(device, torch.long)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)

        self.num_steps = num_steps
        self.step = 0
        self.gae = USE_GAE
        self.gae_tau = gae_tau

    def insert(self, current_obs, action, action_log_probs, value_pred, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        if self.gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * self.gae_tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"* number of steps ({num_steps}) = {num_processes * num_steps} "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch})."
        )
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indicies in sampler:
            observations_batch = self.observations[:-1].view(-1, *self.observations.size()[2:])[indicies]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indicies]
            return_batch = self.returns[:-1].view(-1, 1)[indicies]
            masks_batch = self.masks[:-1].view(-1,1)[indicies]
            old_action_log_probs_batch = self.action_log_probs.view(-1,1)[indicies]
            adv_targ = advantages.view(-1,1)[indicies]

            yield observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

class Model(object):
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__(static_policy, env, config)

        self.env = env
        self.device = config['device']
        self.lr = config['LR']

        self.num_agents = config["num_agents"]
        self.num_actions = env.action_space.n
        self.num_feats = env.observation_space.shape
        self.num_feats = (self.num_feats[0] * 4, *self.num_feats[1:])
        self.value_loss_weight = config['value_loss_weight']
        self.entropy_loss_weight = config['entropy_loss_weight']
        self.rollout = config['rollout']
        self.grad_norm_max = config['grad_norm_max']

        self.ppo_epoch = config['ppo_epoch']
        self.num_mini_batch = config['num_mini_batch']
        self.clip_param = config['ppo_clip_param']

        self.optimzer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)

        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
                                       self.num_feats, self.env.action_space, self.device, config["USE_GAE"],
                                       config['gae_tau'])

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []
        self.losses = []

    def declare_networks(self):
        self.model = ActorCritic(self.num_agents, self.num_actions)

    def compute_loss(self, sample):
        observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

        values, action_log_probs, dist_entropy = self.evaluate_actions(observations_batch, actions_batch)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        # 使用"-"，因为优化器[默认]方式为梯度下降
        actions_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(return_batch, values)

        loss = actions_loss + self.value_loss_weight * value_loss - self.entropy_loss_weight * dist_entropy
        return loss, actions_loss, value_loss, dist_entropy

    def evaluate_actions(self, s, actions):
        logits, values = self.model(s)

        # Categorical()：强化学习中随机策略采样函数，根据probility or logits进行随机采样
        dist = torch.distributions.Categorical(logits=logits)

        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy

    def save_loss(self, loss, policy_loss, value_loss, entropy_loss):
        self.losses.append(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)

    def get_actions(self, s, deterministric=False):
        logits, value = self.model(s)
        dist = torch.distributions.Categorical(logits)
        if deterministric:
            actions = dist.probs.argmax(dim=1, keepdim=True)
        else:
            actions = dist.sample().view(1, -1)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        return value, actions, action_log_probs

    def update(self, rollout):
        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 0.5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollout.feed_forward_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                loss, action_loss, value_loss, dist_entropy = self.compute_loss(sample)

                self.optimzer.zero_grad()
                loss.backward()
                #  torch.nn.utils.clip_grad_norm_梯度裁剪(用于解决梯度消失和梯度爆炸问题)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
                self.optimzer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        value_loss_epoch /= (self.ppo_epoch * self.num_mini_batch)
        action_loss_epoch /= (self.ppo_epoch * self.num_mini_batch)
        dist_entropy_epoch /= (self.ppo_epoch * self.num_mini_batch)
        total_loss = value_loss_epoch + action_loss_epoch + dist_entropy_epoch

        self.save_loss(total_loss, action_loss_epoch, value_loss_epoch, dist_entropy_epoch)
        return action_loss_epoch, value_loss_epoch, dist_entropy_epoch


if __name__ == '__main__':
    seed = 1

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.set_num_threads(1)

    env_id = "PongNoFrameskip-v4"
    envs = [make_env_a2c_atari(env_id, seed, i, log_dir) for i in range(config['num_agents'])]
    envs = SubsetproVecEnv(envs) if config['num_agents'] > 1 else DummyVecEnv(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0]*4, *obs_shape[1:])

    model = Model(env=envs, config=config)

    current_obs = torch.zeros(config['num_agents'], *obs_shape,
                              device=config['device'], dtype=torch.float)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs.astype(np.float32)).to(config["device"])
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    model.rollouts.observations[0].copy_(current_obs)
    episode_rewards = np.zeros(config["num_agents"], dtype=np.float32)
    final_rewards = np.zeros(config['num_agents'], dtype=np.float32)

    start = timer()

    print_step = 1
    print_threshold = 10

    for frame_idx in range(1, config["MAX_FRAMES"] + 1):
        for step in range(config['rollout']):
            with torch.no_grad():
                values, actions, action_log_prob = model.get_actions(model.rollouts.observations[step])
            cpu_actions = actions.view(-1).cpu().numpy()

            obs, reward, done, _ = envs.step(cpu_actions)

            episode_rewards += reward
            masks = 1. - done.astype(np.float32)
            final_rewards *= masks
            final_rewards += (1. - masks) * episode_rewards
            episode_rewards *= masks

            rewards = torch.from_numpy(reward.astype(np.float32)).view(-1, 1).to(config["device"])
            masks = torch.from_numpy(masks).to(config["device"]).view(-1,1)

            current_obs *= masks.view(-1, 1, 1, 1)
            update_current_obs(obs)

            model.rollouts.insert(current_obs, actions.view(-1, 1), action_log_prob, values, rewards, masks)

        with torch.no_grad():
            next_value = model.get_values(model.rollouts.observations[-1])

        model.rollouts.compute_returns(next_value, config['GAMMA'])

        value_loss, action_loss, dist_entropy = model.update(model.rollouts)
        model.rollouts.after_update()

    model.save_w()
    envs.close()