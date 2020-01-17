import sys
import torch
import random
import numpy as np
from param import config
from environment.rl_interface import RLEnv
from torch.distributions import Categorical
from utils.state_transform import get_tensors
from gcn.batch_mgcn_policy import Batch_MGCN_Policy


def test_model(policy_net):
    env = RLEnv()
    state = env.reset()
    node_feats, adj_mats, switch_mask = state
    done = False
    while not done:
        node_feats_torch, adj_mats_torch, switch_mask_torch = \
            get_tensors(node_feats, adj_mats, switch_mask)

        switch_log_pi, switch_pi, masked_pi = policy_net(
            node_feats_torch, adj_mats_torch, switch_mask_torch)

        # sample action
        switch_p = Categorical(masked_pi)
        switch_a = switch_p.sample().item()

        next_state, reward, done = env.step(switch_a)
        node_feats, adj_mats, switch_mask = next_state

        print(switch_a, reward)


if __name__ == '__main__':
    n_epoch = sys.argv[1]

    np.random.seed(config.seed)
    random.seed(config.seed)

    policy_net = Batch_MGCN_Policy(
        config.num_switches, [config.class_feat, config.path_feat,
        config.link_feat, config.switch_feat], config.n_output,
        config.hid_dim, config.h_size, config.n_steps)
    state_dicts = torch.load(config.result_folder + "policy_net_epoch_{}".format(n_epoch))
    policy_net.load_state_dict(state_dicts)

    test_model(policy_net)
