# core framework follows https://github.com/tkipf/pygcn

import numpy as np
import torch
import torch.nn as nn
from gcn.layers import FullyConnectNN
from gcn.mmsg import MultiMessagePassing


class MGCN(nn.Module):
    def __init__(self, n_feats, n_output, n_hids, h_size, n_steps,
                 act=nn.LeakyReLU, layer_norm_on=False):
        '''
        n_feats: number of node features (a list)
        n_output: output size of node
        n_hids: number of hidden neurons (a list)
        n_steps: number of message passing steps
        
        The list in the input features correspond to
        different type of the intput. For each type,
        there are 3 processing units:

        1. map all raw feature to same (high) dimension
        2. individually process features from its type
        3. aggregate based on corresponding adjacency matrix
        '''

        super(MGCN, self).__init__()

        # raise dimension for input vectors
        self.raw = [FullyConnectNN(i, n_hids, h_size, act,
                    layer_norm_on) for i in n_feats]
        # message passing step
        self.msg = MultiMessagePassing(
            [h_size for _ in range(len(n_feats))], n_hids,
            h_size, n_steps, act, layer_norm_on)
        # map the output to desire dimension
        self.transform = FullyConnectNN(h_size, n_hids, n_output,
            act, layer_norm_on)

    def forward(self, node_feats, adj_mats):
        '''
        node_feats: node features, list of size n * d
        adj_mat: adjacancy matrix, list size n * n
        where list len is number of types
        '''

        # raise dimension for input vectors
        in_vec = [r(n) for (n, r) in zip(node_feats, self.raw)]
        # message passing step
        out_vec = self.msg(in_vec, adj_mats)
        # map the output to desire dimension
        node_output = self.transform(out_vec)

        return node_output
