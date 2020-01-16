from environment.rl_interface import RLEnv
from utils.state_transform import get_tensors


class TrajectoryGenerator(object):
    def __init__(self):
        self.env = RLEnv()

    def reset(self):
        state = self.env.reset()
        node_feats, adj_mats, switch_mask = state
        return node_feats, adj_mats, switch_mask

    def step(self, action):
        next_state, reward, done = self.env.step(action)

        if done:
            # automatically reset the environment
            next_state = self.reset()

        node_feats, adj_mats, switch_mask = next_state
        
        return node_feats, adj_mats, switch_mask, \
               reward, done
