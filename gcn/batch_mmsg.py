import torch
import torch.nn as nn
from gcn.mmsg import MultiMessagePassing


class BatchMultiMessagePassing(MultiMessagePassing):
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
        super(BatchMultiMessagePassing, self).__init__(
            in_feats, n_hids, out_feats, num_steps, act, layer_norm_on)


    def forward(self, in_vec, adj_mats):
        # assume in_vec in shape list(batch_size * num_nodes *
        # num_hid_features) where the list len is number of types
        assert(len(in_vec) == self.num_types)
        
        # get batch size and feature size
        ba_size = in_vec[0].shape[0]
        fe_size = in_vec[0].shape[2]

        x = in_vec
        # merge the list, adj_mat will take care
        # of the rest (but keep in mind that "0" has
        # concrete meanings)
        x = torch.stack(x, dim=0).sum(dim=0)
        
        # reshape to 2D
        x = x.reshape([-1, fe_size])

        for s in range(self.num_steps):
            # transform each feature vector
            ns = [f(x) for f in self.fs]
            # reshape to 3D
            ns = [n.reshape([ba_size, -1, fe_size]) for n in ns]
            # batch message passing for all types
            ns = [torch.bmm(adj_mat, n) for (adj_mat, n) in zip(adj_mats, ns)]
            # merge based for all types
            ns = torch.stack(ns, dim=0).sum(dim=0)
            # reshape to 2D
            ns = ns.reshape([-1, fe_size])
            # aggregate feature vectors
            ns = self.g(ns)
            # add back original features
            x = x + ns

        # reshape to 3D
        x = x.reshape([ba_size, -1, fe_size])

        return x
