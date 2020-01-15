import torch


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