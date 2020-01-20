import unittest
from environment.rl_interface import RLEnv
from utils.state_transform import get_tensors


class TestEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_env_init(self):
        env = RLEnv()
        env.reset()

    def test_env_an_episode(self):
        env = RLEnv()
        state = env.reset()
        node_feats_torch, adj_mats_torch, switch_mask_torch = \
            get_tensors(*state)
        for (i, s) in enumerate(switch_mask_torch.numpy()[0]):
            if s:
                state, reward, done = env.step(i)

        node_feats_torch, adj_mats_torch, switch_mask_torch = \
            get_tensors(*state)
        assert(sum(switch_mask_torch.numpy()[0]) == 1)

    def test_env_rewards_manual_actions(self):
        env = RLEnv()
        state = env.reset()
        num_switches = env.num_switches
        switches_to_update = set(env.switches_to_update)
        rewards = []
        actions = []
        done_switch_set = set()
        left_switch_set = set(switches_to_update)
        remaining_switches = []
        done = False
        while not done: 
            print("input switch to update: ")
            switch_a = int(input())
            next_state, reward, done = env.step(switch_a)

            actions.append(switch_a)
            rewards.append(reward)
            done_switch_set.add(switch_a)

            remaining_switches.append(left_switch_set)

            if switch_a == num_switches:
                left_switch_set = switches_to_update - done_switch_set

        print(rewards)
        agg_reward = 0.0
        num_steps_left = 1
        for i in reversed(range(len(rewards))):
            agg_reward = rewards[i] + agg_reward
            if actions[i] == num_switches:
                num_steps_left = num_steps_left + 1
                print(remaining_switches[i], num_steps_left, agg_reward)
