import unittest
import torch
from gcn.layers import FullyConnectNN


class TestFullyConnectedLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        mlp = FullyConnectNN(8, [16, 32, 16], 4)
        assert(len(mlp.sequential) == 8)

    def test_init_final_layer_no_act(self):
        mlp = FullyConnectNN(8, [16, 32, 16], 4,
                             final_layer_act=False)
        assert(len(mlp.sequential) == 7)

    def test_init_layer_norm(self):
        mlp = FullyConnectNN(8, [16, 32, 16], 4,
                             layer_norm=True)
        assert(len(mlp.sequential) == 12)

    def test_feed_forward(self):
        mlp = FullyConnectNN(8, [16, 32, 16], 4)
        in_vec = torch.FloatTensor(1, 8)
        mlp(in_vec)

    def test_feed_forward_batch(self):
        mlp = FullyConnectNN(8, [16, 32, 16], 4)
        in_vec = torch.FloatTensor(12, 8)
        mlp(in_vec)

    def test_feed_forward_batch_correctness(self):
        mlp = FullyConnectNN(3, [16, 32, 16], 4)
        in_vec = torch.FloatTensor([[1,2,3], [1,2,3]])
        out_vec = mlp(in_vec)
        assert(out_vec.shape == torch.Size([2, 4]))
        assert(torch.all(out_vec[0, :] == out_vec[1, :]).item())
