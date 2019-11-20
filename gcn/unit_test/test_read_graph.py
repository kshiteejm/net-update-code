import unittest
import torch
import torch.nn as nn
import numpy as np
from gcn.mgcn import MGCN
from gcn.layers import FullyConnectNN


class TestReadGraph(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node_features = np.load(
            './rust-dp/data/nodes_datum.npy')
        self.adj_mats = np.load(
            './rust-dp/data/adj_mat_datum.npy')
        self.node_feats_torch = [torch.FloatTensor(nf) \
            for nf in self.node_features]
        self.adj_mats_torch = [torch.FloatTensor(adj) \
            for adj in self.adj_mats]
        self.l2_loss = torch.nn.MSELoss(reduction='mean')


    def test_graph_read(self):
        mgcn = MGCN(n_feats=[2, 2, 2, 2],
                    n_output=8,
                    n_hids=[16, 32],
                    h_size=8,
                    n_steps=8)

        gcn_out = mgcn(self.node_feats_torch,
                       self.adj_mats_torch)

    def test_map_one_value(self):
        mgcn = MGCN(n_feats=[2, 2, 2, 2],
                    n_output=8,
                    n_hids=[16, 32],
                    h_size=8,
                    n_steps=8)  # layer_norm_on = False

        gcn_out = mgcn(self.node_feats_torch,
                       self.adj_mats_torch)

        # assume switch is 0-19
        switch_idx = np.ones([20, 8])
        for i in range(20):
            switch_idx[i, :] = i
        switch_idx = torch.LongTensor(switch_idx)

        # assume ground truth is 4
        switch_out_target = torch.FloatTensor([4])

        # per switch gcn output to hid
        f = FullyConnectNN(8, [16, 16], 8)

        # map aggregated hid to single number
        g = FullyConnectNN(8, [16], 1)

        # gather switch output
        switch_out = gcn_out.gather(0, switch_idx)

        # transform the output
        switch_out = f(switch_out)

        # aggregate
        switch_out = torch.sum(switch_out, dim=0)

        # final transform
        switch_out = g(switch_out)

        # l2 loss
        loss = self.l2_loss(switch_out, switch_out_target)

        # backward
        loss.backward()

    def test_map_one_value_with_gradient_update(self):
        class MGCN_Value(nn.Module):
            def __init__(self):
                super(MGCN_Value, self).__init__()
                self.mgcn = MGCN(n_feats=[2, 2, 2, 2],
                            n_output=8,
                            n_hids=[16, 32],
                            h_size=8,
                            n_steps=8)  # layer_norm_on = False
                self.f_gcn_out = FullyConnectNN(8, [16, 16], 8)
                self.g_gcn_out = FullyConnectNN(8, [16], 1)

            def forward(self, node_feats, adj_mats):
                gcn_out = self.mgcn(node_feats, adj_mats)

                # assume switch is 0-19
                switch_idx = np.ones([20, 8])
                for i in range(20):
                    switch_idx[i, :] = i
                switch_idx = torch.LongTensor(switch_idx)

                # gather switch output
                switch_out = gcn_out.gather(0, switch_idx)

                # transform the output
                switch_out = self.f_gcn_out(switch_out)

                # aggregate
                switch_out = torch.sum(switch_out, dim=0)

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
        switch_out_target = torch.FloatTensor([4])

        # l2 loss
        loss = self.l2_loss(switch_out, switch_out_target)

        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()

