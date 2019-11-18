import numpy as np
import torch
import torch.nn as nn
from gcn.layers import FullyConnectNN
from gcn.mmsg import MultiMessagePassing
from gcn.mgcn import MGCN


class MGCN_Value(nn.Module):
    def __init__(self, n_switches, n_feats, n_output, n_hids, h_size, n_steps,
                 act=nn.LeakyReLU, layer_norm_on=False):
        super(MGCN_Value, self).__init__()

        self.n_switches = n_switches
        self.n_output = n_output

        self.mgcn = MGCN(n_feats=n_feats,
                    n_output=n_output,
                    n_hids=n_hids,
                    h_size=h_size,
                    n_steps=n_steps,
                    act=act,
                    layer_norm_on=layer_norm_on)
        
        # per switch gcn output to hid
        self.f_gcn_out = FullyConnectNN(n_output, [16, 16], n_output)
        
        # map aggregated hid to single number
        self.g_gcn_out = FullyConnectNN(n_output, [16], 1)


    def forward(self, node_feats, adj_mats):
        gcn_out = self.mgcn(node_feats, adj_mats)

        switch_idx = np.ones([self.n_switches, self.n_output])
        for i in range(self.n_switches):
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