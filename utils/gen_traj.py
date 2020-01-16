from environment.rl_interface import RLEnv
from utils.state_transform import get_tensors


class TrajectoryGenerator(object):
    def __init__(self):
        self.env = RLEnv()

    def reset(self):
        state = self.env.reset()
        node_feats_torch, adj_mats_torch, switch_mask_torch = \
            get_tensors(state)
        return node_feats_torch, adj_mats_torch, switch_mask_torch

    def step(self, action):
        next_state, reward, done = env.step(action)

        if done:
            # automatically reset the environment
            node_feats_torch, adj_mats_torch, switch_mask_torch = \
                self.reset()
        else:    
            node_feats_torch, adj_mats_torch, switch_mask_torch = \
                get_tensors(next_state)

        return node_feats_torch, adj_mats_torch, switch_mask_torch, \
               reward, done
