import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from gcn.batch_mgcn import BatchMGCN
from gcn.layers import FullyConnectNN


class Batch_MGCN_Policy(nn.Module):
    def __init__(self, n_switches, n_feats, n_output, n_hids, h_size, n_steps,
                 act=nn.LeakyReLU, layer_norm_on=False):
        super(Batch_MGCN_Policy, self).__init__()

        self.n_switches = n_switches
        self.n_output = n_output

        self.mgcn = BatchMGCN(n_feats=n_feats,
                              n_output=n_output,
                              n_hids=n_hids,
                              h_size=h_size,
                              n_steps=n_steps,
                              act=act,
                              layer_norm_on=layer_norm_on)

        self.f_gcn_out = FullyConnectNN(n_output, n_hids, n_output)
        self.priority_transform = FullyConnectNN(n_output, n_hids, 1,
            final_layer_act=False)
        self.agg_priority_transform = FullyConnectNN(n_output, n_hids, 1,
            final_layer_act=False)

    def forward(self, node_feats, adj_mats, mask):
        gcn_out = self.mgcn(node_feats, adj_mats)

        batch_size = node_feats[0].shape[0]

        switch_idx = np.ones([batch_size, self.n_switches, self.n_output])
        for i in range(self.n_switches):
            switch_idx[:, i, :] = i
        switch_idx = torch.LongTensor(switch_idx)

        # gather switch output
        switch_out = gcn_out.gather(1, switch_idx)

        # reshape to 2D
        switch_out = torch.reshape(switch_out,
            [batch_size * self.n_switches, -1])

        # transform the output
        switch_out = self.f_gcn_out(switch_out)

        # map to single priority value (batch_size * n_switches, 1)
        switch_priority = self.priority_transform(switch_out)

        # reshape back to 2D (batch_size, n_switches)
        switch_priority = torch.reshape(switch_priority,
            [batch_size, self.n_switches])

        # reshape back to 3D
        switch_out = torch.reshape(switch_out,
            [batch_size, self.n_switches, -1])

        # aggregate
        agg_switch_out = torch.sum(switch_out, dim=1)

        # map to single priority value for the terminal action
        term_priority = self.g_gcn_out(agg_switch_out)

        # reshape to 2D (batch_size, 1)
        term_priority = torch.reshape(term_priority,
            [batch_size, 1])

        # concatenate switch priority with terminal priority
        priority_values = torch.cat([
            switch_priority, term_priority], dim=1)

        # output log pi (softmax and smaple from outside)
        log_pi = F.log_softmax(priority_values, dim=-1)

        return log_pi
