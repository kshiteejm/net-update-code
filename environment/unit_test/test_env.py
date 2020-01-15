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
        node_feats_torch, adj_mats_torch, switch_mask_torch = get_tensors(state)
        for (i, s) in enumerate(switch_mask_torch.numpy()[0]):
            if s:
                state, reward, done = env.step(i)

        node_feats_torch, adj_mats_torch, switch_mask_torch = get_tensors(state)
        assert(sum(switch_mask_torch.numpy()[0]) == 1)
