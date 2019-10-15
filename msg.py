# source from https://github.com/hongzimao/gcn_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import layers

class OneWayMessagePassing(nn.Module):
    def __init__(self, transform):
        '''
        feats: number of features
        n_hids: number of hidden neurons (a list)
        g performs message passing transformations
        y_i <- g(sum_neighbors_j y_j), at every step
        '''
        super(OneWayMessagePassing, self).__init__()

        self.g = transform

    def forward(self, in_vec, adj_mat):
        ne = torch.mm(adj_mat, in_vec)  # message passing
        ne = self.g(ne)

        return ne