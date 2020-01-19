import os
import sys
import torch
import random
import numpy as np
from param import config
from environment.rl_interface import RLEnv
from torch.distributions import Categorical
from utils.state_transform import get_tensors
from gcn.batch_mgcn_value import Batch_MGCN_Value
from gcn.batch_mgcn_policy import Batch_MGCN_Policy


def test_model(policy_net, value_net):
    env = RLEnv()
    state = env.reset()
    node_feats, adj_mats, switch_mask = state
    done = False
    while not done:
        node_feats_torch, adj_mats_torch, switch_mask_torch = \
            get_tensors(node_feats, adj_mats, switch_mask)

        values = value_net(
            node_feats_torch, adj_mats_torch)
        
        switch_log_pi, switch_pi, masked_pi = policy_net(
            node_feats_torch, adj_mats_torch, switch_mask_torch)

        # # sample action
        # switch_p = Categorical(masked_pi)
        # switch_a = switch_p.sample().item()
        
        print(masked_pi)
        print("manually enter action: ")
        switch_a = int(input())

        next_state, reward, done = env.step(switch_a)
        node_feats, adj_mats, switch_mask = next_state

        switch_log_pi, switch_pi, masked_pi = policy_net(
            node_feats_torch, adj_mats_torch, switch_mask_torch)
        
        print(masked_pi)
        print(switch_a, reward)
        print(values.detach().item())

def get_optimal_action():
    env = RLEnv()
    env.reset()
    env.dcn_environment.generate_costs("/tmp/costs.csv")
    os.system("cd ./rust-dp; \
              ./target/debug/rust-dp --num-nodes 20 --num-steps 4 \
              --update-idx 0 1 2 3 4 5 8 9 12 13 16 17 \
              --cm-path /tmp/costs.csv \
              --action-seq-path /tmp/action_seq.csv \
              --action-path /tmp/actions.csv \
              --value-path /tmp/values.csv")
    os.system("cat /tmp/action_seq.csv")

if __name__ == '__main__':
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    policy_net = Batch_MGCN_Policy(
        config.num_switches, [config.class_feat, config.path_feat,
        config.link_feat, config.switch_feat], config.n_output,
        config.hid_dim, config.h_size, config.n_steps)
    
    value_net = Batch_MGCN_Value(
        config.num_switches, [config.class_feat, config.path_feat,
        config.link_feat, config.switch_feat], config.n_output,
        config.hid_dim, config.h_size, config.n_steps)
    
    if config.saved_policy_model is not None:
        policy_net.load_state_dict(torch.load(config.saved_policy_model))

    if config.saved_value_model is not None:
        value_net.load_state_dict(torch.load(config.saved_value_model))

    test_model(policy_net, value_net)
    # get_optimal_action()
