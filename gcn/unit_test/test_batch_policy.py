import unittest
import torch
import torch.nn as nn
import numpy as np
from gcn.batch_mgcn import BatchMGCN
from gcn.layers import FullyConnectNN
from gcn.batch_mgcn_policy import Batch_MGCN_Policy


class TestBatchPolicy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node_features = np.load(
            './rust-dp/data/nodes_datum.npy')
        self.adj_mats = np.load(
            './rust-dp/data/adj_mat_datum.npy')

        self.batch_node_features = [np.tile(n, [32, 1, 1])
            for n in self.node_features]
        self.batch_adj_mats = [np.tile(a, [32, 1, 1])
            for a in self.adj_mats]

        self.node_feats_torch = [torch.FloatTensor(nf) \
            for nf in self.batch_node_features]
        self.adj_mats_torch = [torch.FloatTensor(adj) \
            for adj in self.batch_adj_mats]

        self.mgcn_value = Batch_MGCN_Policy(
            n_switches=20,
            n_feats=[2, 2, 2, 2],
            n_output=8,
            n_hids=[16, 32],
            h_size=8,
            n_steps=8)  # layer_norm_on = False

        self.l2_loss = torch.nn.MSELoss(reduction='mean')


    def test(self):
        pass