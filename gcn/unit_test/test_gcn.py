import unittest
import torch
from gcn.mgcn import MGCN


class TestGraphConvolution(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        mgcn = MGCN([2, 3], 1, [16, 32], 8, 2)
        assert(len(mgcn.raw) == 2)
        assert(len(mgcn.raw[0].sequential) == 6)
        assert(len(mgcn.raw[1].sequential) == 6)
        assert(len(mgcn.msg.g.sequential) == 6)
        assert(len(mgcn.msg.fs) == 2)
        assert(len(mgcn.msg.fs[0].sequential) == 6)
        assert(len(mgcn.msg.fs[1].sequential) == 6)

    def test_feedforward(self):
        mgcn = MGCN([2, 3], 1, [16, 32], 8, 2)
        node_in_1 = torch.FloatTensor([[1,2], [0,0]])
        node_in_2 = torch.FloatTensor([[0,0,0], [1,2,3]])
        adj_mats_1 = torch.FloatTensor([[0, 1], [0, 0]])
        adj_mats_2 = torch.FloatTensor([[0, 0], [1, 0]])
        node_out = mgcn([node_in_1, node_in_2],
                       [adj_mats_1, adj_mats_2])

    def test_feedforward_more_steps(self):
        mgcn = MGCN([2, 3], 1, [16, 32], 8, 6)
        node_in_1 = torch.FloatTensor([[1,2], [0,0]])
        node_in_2 = torch.FloatTensor([[0,0,0], [1,2,3]])
        adj_mats_1 = torch.FloatTensor([[0, 1], [0, 0]])
        adj_mats_2 = torch.FloatTensor([[0, 0], [1, 0]])
        node_out = mgcn([node_in_1, node_in_2],
                       [adj_mats_1, adj_mats_2])


