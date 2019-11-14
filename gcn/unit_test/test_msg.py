import unittest
import torch
from gcn.mmsg import MultiMessagePassing


class TestMessagePassing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        mgcn = MultiMessagePassing([3, 3], [16, 32], 3, 2)
        assert(len(mgcn.g.sequential) == 6)
        assert(len(mgcn.fs) == 2)
        assert(len(mgcn.fs[0].sequential) == 6)
        assert(len(mgcn.fs[1].sequential) == 6)

    def test_init_layer_norm_on(self):
        mgcn = MultiMessagePassing([3, 3], [16, 32], 3, 2,
                                   layer_norm_on=True)
        assert(len(mgcn.g.sequential) == 9)
        assert(len(mgcn.fs) == 2)
        assert(len(mgcn.fs[0].sequential) == 9)
        assert(len(mgcn.fs[1].sequential) == 9)

    def test_feedforward(self):
        mgcn = MultiMessagePassing([3, 3], [16, 32], 3, 2,
                                   layer_norm_on=True)
        in_vec_1 = torch.FloatTensor([[1,2,3], [0,0,0]])
        in_vec_2 = torch.FloatTensor([[0,0,0], [1,2,3]])
        adj_mats_1 = torch.FloatTensor([[0, 1], [0, 0]])
        adj_mats_2 = torch.FloatTensor([[0, 0], [1, 0]])
        out_vec = mgcn([in_vec_1, in_vec_2],
                       [adj_mats_1, adj_mats_2])

