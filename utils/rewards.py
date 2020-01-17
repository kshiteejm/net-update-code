from param import config
import numpy as np


def cumulative_rewards(rewards, dones, gamma,
                       next_values, batch=None):
    l = len(rewards)
    if batch is None:
        # cumulate over multiple batches (at master)
        # need to cut between the boundaries
        batch = l

    returns = np.zeros([l, 1], dtype=np.float32)
    R = next_values[-1]

    for i in reversed(range(l)):
        if i % batch == batch - 1:
            R = next_values[i]
        if dones[i]:
            R = 0
        R = rewards[i] + gamma * R
        returns[i] = R

    return returns


def get_advantage(cum_rewards, values, norm=False):
    adv = cum_rewards - values
    if norm:
        adv = (adv - adv.mean()) / (adv.std() + config.eps)
    return adv


def gae_advantage(rewards, dones, values, next_values,
                  gamma, lam, batch=None, norm=False):
    # TD lambda style advantage computation
    # more details in GAE: https://arxiv.org/pdf/1506.02438.pdf
    l = len(rewards)
    if batch is None:
        # cumulate over multiple batches (at master)
        # need to cut between the boundaries
        batch = l

    adv = np.zeros([l, 1], dtype=np.float32)
    last_gae = 0

    for i in reversed(range(l)):
        if i % batch == batch - 1:
            last_gae = 0

        not_done = 1 - dones[i]

        # TD(0) at each step
        delta = rewards[i] + gamma * next_values[i] \
            * not_done - values[i]

        adv[i] = last_gae = delta + gamma * lam \
            * last_gae * not_done

    if norm:
        adv = (adv - adv.mean()) / (adv.std() + config.eps)

    return adv

def get_monitor_total_rewards(rewards, dones):
    r = None
    monitor_rewards = []
    for i in reversed(range(len(rewards))):
        if dones[i] == 1:
            if r is not None:
                monitor_rewards.append(r)
            r = 0

        if r is not None:
            r += rewards[i]

    if r is not None:
        monitor_rewards.append(r)

    if len(monitor_rewards) == 0:
        monitor_rewards.append(0)

    return monitor_rewards