import unittest
import numpy as np
from param import config
from environment.rl_interface import RLEnv
from utils.state_transform import get_tensors


class TestDenseReward(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_dense_reward(self):
        repeat_exp = config.repeat_exp
        dense_reward = config.dense_reward
        config.repeat_exp = 1  # use same seed

        # first we compute the sparse reward
        config.dense_reward = 0
        env = RLEnv()
        state = env.reset()
        
        # update 3 switches in 1 batch
        three_switches = 3
        done = False
        total_reward = 0
        while not done:
            if three_switches == 0:
                switch_a = env.num_switches
                three_switches = 3
            else:
                switch_a = next(iter(env.switches_to_update - env.intermediate_switches))
                three_switches -= 1

            next_state, reward, done = env.step(switch_a)
            total_reward += reward

        # now try the dense reward
        config.dense_reward = 1
        env = RLEnv()
        state = env.reset()

        # update 3 switches in 1 batch
        three_switches = 3
        done = False
        total_dense_reward = 0
        while not done:
            if three_switches == 0:
                switch_a = env.num_switches
                three_switches = 3
            else:
                switch_a = next(iter(env.switches_to_update - env.intermediate_switches))
                three_switches -= 1

            next_state, reward, done = env.step(switch_a)
            total_dense_reward += reward

        # sum of dense reward is the same as sum of sparse reward
        assert(np.isclose(total_dense_reward, total_reward, config.eps))

        # reset the parameters
        config.repeat_exp = repeat_exp
        config.dense_reward = dense_reward
