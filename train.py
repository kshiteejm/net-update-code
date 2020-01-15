import time
from environment.rl_interface import RLEnv
from gcn.batch_mgcn_policy import Batch_MGCN_Policy
from gcn.batch_mgcn_value import Batch_MGCN_Value
import torch
from torch.distributions import Categorical


def main():
    policy_net = Batch_MGCN_Policy(20, [3,3,3,3], 8, [16,32], 8, 8)
    value_net = Batch_MGCN_Value(20, [3,3,3,3], 8, [16,32], 8, 8)
    env = RLEnv()
    state = env.reset()
    node_feats_torch, adj_mats_torch, switch_mask_torch = get_tensors(state)
    # convert state
    switch_logpi, switch_pi, masked_pi = policy_net(
        node_feats_torch, adj_mats_torch, switch_mask_torch)

    switch_p = Categorical(masked_pi)
    switch_a = switch_p.sample().item()

    env.step(switch_a)

def get_tensors(state):
    nodefeats, adjmats, switch_mask = state
    num_types = 4
    batch_node_feats = []
    batch_adj_mats = []
    batch_switch_mask = []
    for _ in range(num_types):
        batch_node_feats.append([])
        batch_adj_mats.append([])
    for type_i in range(num_types):
        batch_node_feats[type_i].append(nodefeats[type_i])
        batch_adj_mats[type_i].append(adjmats[type_i])
    batch_switch_mask.append(switch_mask)
    node_feats_torch = [torch.FloatTensor(nf) \
                        for nf in batch_node_feats]
    adj_mats_torch = [torch.FloatTensor(adj) \
                      for adj in batch_adj_mats]
    switch_mask_torch = torch.FloatTensor(batch_switch_mask)
    return node_feats_torch, adj_mats_torch, switch_mask_torch

if __name__ == '__main__':
    main()
