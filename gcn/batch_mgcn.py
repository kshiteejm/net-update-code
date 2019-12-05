# core framework follows https://github.com/tkipf/pygcn

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleList
from gcn.layers import FullyConnectNN
from gcn.batch_mmsg import BatchMultiMessagePassing


class BatchMGCN(nn.Module):
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

        super(BatchMGCN, self).__init__()

        # raise dimension for input vectors
        self.raw = ModuleList([FullyConnectNN(i, n_hids, h_size, act,
                    layer_norm_on) for i in n_feats])
        # message passing step
        self.msg = BatchMultiMessagePassing(
            [h_size for _ in range(len(n_feats))], n_hids,
            h_size, n_steps, act, layer_norm_on)
        # map the output to desire dimension
        self.transform = FullyConnectNN(h_size, n_hids, n_output,
            act, layer_norm_on)

    def forward(self, node_feats, adj_mats):
        '''
        node_feats: list of batch * num_nodes * num_features
        adj_mat: list of batch * num_nodes * num_nodes
        where list len is number of types
        '''

        batch_size = node_feats[0].shape[0]
        num_nodes = node_feats[0].shape[1]
        feature_size = node_feats[0].shape[2]
        # reshape to 2D
        node_feats = [n.reshape([-1, feature_size])
            for n in node_feats]
        # raise dimension for input vectors
        in_vec = [r(n) for (n, r) in zip(node_feats, self.raw)]
        # reshape back to 3D
        in_vec = [i.reshape([batch_size, num_nodes, -1])
            for i in in_vec]
        # message passing step
        out_vec = self.msg(in_vec, adj_mats)
        # reshape to 2D
        out_vec = out_vec.reshape([batch_size * num_nodes, -1])
        # map the output to desire dimension
        node_output = self.transform(out_vec)
        # reshape back to 3D
        node_output = node_output.reshape([batch_size, num_nodes, -1])

        return node_output
