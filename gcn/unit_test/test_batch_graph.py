import unittest
import torch
import torch.nn as nn
import numpy as np
from gcn.batch_mgcn import BatchMGCN
from gcn.layers import FullyConnectNN


class TestBatchGraph(unittest.TestCase):
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

        self.l2_loss = torch.nn.MSELoss(reduction='mean')


    def test_map_one_value_with_batch_gradient_update(self):
        class MGCN_Value(nn.Module):
            def __init__(self):
                super(MGCN_Value, self).__init__()
                self.mgcn = BatchMGCN(n_feats=[2, 2, 2, 2],
                            n_output=8,
                            n_hids=[16, 32],
                            h_size=8,
                            n_steps=8)  # layer_norm_on = False
                self.f_gcn_out = FullyConnectNN(8, [16, 16], 8)
                self.g_gcn_out = FullyConnectNN(8, [16], 1)

            def forward(self, node_feats, adj_mats):
                gcn_out = self.mgcn(node_feats, adj_mats)

                # assume switch is 0-19
                switch_idx = np.ones([32, 20, 8])
                for i in range(20):
                    switch_idx[:, i, :] = i
                switch_idx = torch.LongTensor(switch_idx)

                # gather switch output
                switch_out = gcn_out.gather(1, switch_idx)

                # get the shape
                batch_size = switch_out.shape[0]
                num_switches = switch_out.shape[1]

                # reshape to 2D
                switch_out = torch.reshape(switch_out,
                    [batch_size * num_switches, -1])

                # transform the output
                switch_out = self.f_gcn_out(switch_out)

                # reshape back to 3D
                switch_out = torch.reshape(switch_out,
                    [batch_size, num_switches, -1])

                # aggregate
                switch_out = torch.sum(switch_out, dim=1)

                # final transform
                switch_out = self.g_gcn_out(switch_out)

                return switch_out

        # put everything end-to-end into a module so that we can call
        # .parameters() for the optimizer
        mgcn_value = MGCN_Value()

        opt = torch.optim.Adam(mgcn_value.parameters(), lr=1e-3)

        switch_out = mgcn_value(self.node_feats_torch,
                                self.adj_mats_torch)

        # assume ground truth is 4
        switch_out_target = torch.FloatTensor([[4] for _ in range(32)])

        # l2 loss
        loss = self.l2_loss(switch_out, switch_out_target)

        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()

