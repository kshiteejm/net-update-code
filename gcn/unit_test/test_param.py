import unittest
import torch
import numpy as np
from gcn.batch_mgcn_value import Batch_MGCN_Value


class TestParams(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_same_param(self):
        mgcn_1 = Batch_MGCN_Value(
            n_switches=20,
            n_feats=[2, 2, 2, 2],
            n_output=8,
            n_hids=[16, 32],
            h_size=8,
            n_steps=1, 
            layer_norm_on=True)

        mgcn_2 = Batch_MGCN_Value(
            n_switches=20,
            n_feats=[2, 2, 2, 2],
            n_output=8,
            n_hids=[16, 32],
            h_size=8,
            n_steps=1, 
            layer_norm_on=True)

        mgcn_3 = Batch_MGCN_Value(
            n_switches=20,
            n_feats=[2, 2, 2, 2],
            n_output=8,
            n_hids=[16, 32],
            h_size=8,
            n_steps=1, 
            layer_norm_on=True)
        
        p = True
        for (p1, p2) in zip(mgcn_2.parameters(), mgcn_3.parameters()):
            if (p1.detach().numpy() == p2.detach().numpy()).all() == False:
                p = False
                break
        assert(p == False)  # parameters should be different

        torch.save(mgcn_1.state_dict(), '/tmp/test_tmp.pt')

        state_dicts = torch.load('/tmp/test_tmp.pt')
        mgcn_2.load_state_dict(state_dicts)
        mgcn_3.load_state_dict(state_dicts)

        p = True
        for (p1, p2) in zip(mgcn_2.parameters(), mgcn_3.parameters()):
            if (p1.detach().numpy() == p2.detach().numpy()).all() == False:
                p = False
                break
        assert(p)  # now the parameters are the same

        node_feats = [torch.FloatTensor(np.random.rand(
            30, 50, 2)) for _ in range(4)]
        adj_mats = [torch.FloatTensor(np.random.randint(
            0, 2, [30, 50, 50])) for _ in range(4)]

        est_1 = mgcn_1(node_feats, adj_mats)
        est_11 = mgcn_1(node_feats, adj_mats)
        est_2 = mgcn_2(node_feats, adj_mats)
        est_3 = mgcn_3(node_feats, adj_mats)

        assert(all(est_1 == est_11))
        assert(all(est_1 == est_2))
        assert(all(est_1 == est_3))
