import torch
import torch.nn as nn
from torch.nn import ModuleList
from gcn.layers import FullyConnectNN


class MultiMessagePassing(nn.Module):
    def __init__(self, in_feats, n_hids, out_feats, num_steps,
                 act=nn.LeakyReLU, layer_norm_on=False):
        '''
        in_feats: number of input features
        n_hids: number of hidden neurons (a list)
        out_feats: number of output features

        f maps *all* raw feature vectors to size out_feats
        g performs message passing transformations

        y_i = f(x_i), for all i
        y_i <- g(sum_neighbors_j y_j) + y_i, at every step
    
        Note: the sum_neightbors applies to all types
        '''
        super(MultiMessagePassing, self).__init__()

        self.num_steps = num_steps
        self.num_types = len(in_feats)

        self.fs = ModuleList([FullyConnectNN(in_feat, n_hids, out_feats,
            act, layer_norm_on) for in_feat in in_feats])
        self.g = FullyConnectNN(out_feats, n_hids, out_feats,
            act, layer_norm_on)

    def forward(self, in_vec, adj_mats):
        # assume in_vec in shape list(num_nodes *
        # num_hid_features) where the list len
        # is number of types
        assert(len(in_vec) == self.num_types)
        x = in_vec
        # merge the list, adj_mat will take care
        # of the rest (but keep in mind that "0" has
        # concrete meanings)
        x = torch.stack(x, dim=0).sum(dim=0)

        for s in range(self.num_steps):
            # transform each feature vector
            ns = [f(x) for f in self.fs]
            # message passing for all types
            ns = [torch.mm(adj_mat, n) for (adj_mat, n) in zip(adj_mats, ns)]
            # merge based for all types
            ns = torch.stack(ns, dim=0).sum(dim=0)
            # aggregate feature vectors
            ns = self.g(ns)
            # add back original features
            x = x + ns

        return x
