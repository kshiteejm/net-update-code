import torch
import torch.nn as nn
import numpy as np
from gcn.batch_mgcn import BatchMGCN
from gcn.layers import FullyConnectNN


class Batch_MGCN_Value(nn.Module):
    def __init__(self, n_feats, n_output, n_hids, h_size, n_steps,
                 act=nn.LeakyReLU, layer_norm_on=False):
        super(Batch_MGCN_Value, self).__init__()
        self.mgcn = BatchMGCN(n_feats=n_feats,
                              n_output=n_output,
                              n_hids=n_hids,
                              h_size=h_size,
                              n_steps=n_steps)

        self.f_gcn_out = FullyConnectNN(n_output, n_hids, n_output)
        self.g_gcn_out = FullyConnectNN(n_output, n_hids, 1,
            final_layer_act=False)

    def forward(self, node_feats, adj_mats, switch_idx):
        gcn_out = self.mgcn(node_feats, adj_mats)

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
