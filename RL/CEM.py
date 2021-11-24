# -*- codeing = utf-8 -*-
# @Time : 2021/11/24 18:19
# @Author : Evan_wyl
# @File : CEM.py

import numpy as np
import gym
from gym import logger
import argparse
from _policies import BinaryActionLinearPolicy

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    n_elite = int(np.round(batch_size * elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in th_std[None, :] * np.random.randn(batch_size, th_mean.size)])
        ys = np.array(f(th) for th in ths)
        elite_ind = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_ind]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean': ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    t = 0
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t % 3 == 0:
            env.render()
        if done:
            break
    return total_rew, t + 1


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('target', nargs="?", default='CartPole-v0')
    args = parser.parse_args()

    env = gym.make(args.target)
    env.seed(0)
    np.random.seed(0)
    params = dict(n_iter=100, batch_size=10, elite_frac=0.2)
    num_steps = 200

    def noisy_evaluation(theta):
        agent = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    for i, iterdata in enumerate(cem(noisy_evaluation, np.zeros(env.observation_space.shape[0] + 1), **params)):
        print("Iteration %2i. Episode mean reward: %7.3f" % (i, iterdata['y_mean']))
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        do_rollout(agent, env, num_steps)
    env.close()
