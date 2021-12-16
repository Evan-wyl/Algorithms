# -*- codeing = utf-8 -*-
# @Time : 2021/12/10 14:41
# @Author : Evan_wyl
# @File : A3C.py

import numpy as np
import math
import os, gym, time, glob, argparse, sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch import optim
from torch import multiprocessing as mp

from scipy.signal import lfilter
from scipy.misc import imresize

os.environ['OPM_NUM_THREADS'] = 1


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn-steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser


# lfilter：使用双二阶滤波对数据进行过滤
discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]
prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end);
    f = open(args.save_dir + 'log.txt', mode);
    f.write(s + '\n');
    f.close()


def normalized_columns_initializer(weight, std=1.0):
    out = torch.randn(weight.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4] * weight_shape[0])
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weights_shape = list(m.weight.data.size())
        fan_in = weights_shape[1]
        fan_out = weights_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_input, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_input, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.GRUCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx = self.lstm(x, hx)
        x = hx
        return self.critic_linear(x), self.actor_linear(x), hx


class SharedAdam(torch.optim.Adam):  # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['shared_steps'] += 1
                self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1  # a "step += 1"  comes later
        super.step(closure)


def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    rewards[-1] += args.gamma * np.values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = 0.5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum()
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_models, shared_optimizer, rank, args, info):
    env = gym.make(args.env)
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    model = ActorCritic(1, args.num_actions)
    state = torch.tensor(prepro(env.reset()))

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True
    hx = torch.zeros(1, 256)
    while info['frames'][0] <= 8e7 or args.test:
        model.load_state_dict(shared_models.state_dict())

        hx = torch.zeros(1, 256) if done else hx.detach()
        values, logps, actions, rewards = [], [], [], []

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model((state.view(1, 1, 80, 80)), hx)
            logp = F.log_softmax(logit, dim=-1)
            action = torch.exp(logp).multinomial(num_samples=1)
            state, reward, done, _ = env.step(action.numpy()[0])
            if args.render: env.render()

            state = torch.tensor(prepro(state))
            epr += reward
            reward = np.clip(reward, -1, 1)
            done = done or episode_length >= 1e4

            info['frames'].add_(1)
            num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0:
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames / 1e6))
                torch.save(shared_models.state_dict(), args.save_dir + 'model.{:.0f}.tar'.format(num_frames / 1e6))

            if done:  # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60:
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss{:.2f}'.format(
                    elapsed, info['episodes'].item(), num_frames / 1e6, info['run_epr'].item(), info['run_loss'].item()
                ))
                last_disp_time = time.time()

            if done:
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value), logps.append(logp), actions.append(action), rewards.append(reward)

        next_value = torch.zeros(1, 1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, values, logps, actions, rewards)
        eploss += loss.item()
        shared_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_models.parameters()):
            if shared_param.grad is None: shared_param.grad = param.grad
        shared_optimizer.step()


if __name__ == '__main__':
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise Exception("Must be using Python 3 with linux!")

    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower())
    if args.render:
        args.processes = 1
        args.test = True

    if args.test:
        args.lr = 0

    args.num_actions = gym.make(args.env).action_space.n
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None

    torch.manual_seed(args.seed)
    shared_model = ActorCritic(1, args.num_actions)
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k:torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0:
        printlog(args, '', end='', mode='w')

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()