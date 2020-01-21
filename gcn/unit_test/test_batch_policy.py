import unittest
import torch
import torch.nn as nn
import numpy as np
from gcn.batch_mgcn import BatchMGCN
from gcn.layers import FullyConnectNN
from torch.distributions import Categorical
from gcn.batch_mgcn_policy import Batch_MGCN_Policy


class TestBatchPolicy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node_features = np.load(
            './rust-dp/data/nodes_datum.npy')
        self.adj_mats = np.load(
            './rust-dp/data/adj_mat_datum.npy')

        self.batch_node_features = [np.tile(n, [32, 1, 1])
            for n in self.node_features]
        self.batch_adj_mats = [np.tile(a, [32, 1, 1])
            for a in self.adj_mats]

        self.node_feats_torch = [torch.FloatTensor(nf) \
            for nf in self.batch_node_features]
        self.adj_mats_torch = [torch.FloatTensor(adj) \
            for adj in self.batch_adj_mats]
        mask_done = False
        while not mask_done:
            # each row should have at least one mask-in value
            mask = np.random.randint(2, size=(32, 21))
            mask_done = all(np.sum(mask, axis=1))
        self.switch_mask = torch.FloatTensor(mask)

        self.mgcn_policy = Batch_MGCN_Policy(
            n_switches=20,
            n_feats=[2, 2, 2, 2],
            n_output=8,
            n_hids=[16, 32],
            h_size=8,
            n_steps=8)  # layer_norm_on = False

        self.l2_loss = torch.nn.MSELoss(reduction='mean')

        self.entropy_factor = 0.1


    def test_policy_feedforwarding(self):
        switch_log_pi, switch_pi, masked_log_pi, masked_pi = self.mgcn_policy(
            self.node_feats_torch, self.adj_mats_torch, self.switch_mask)
        assert(switch_log_pi.shape[0] == 32)  # batch_size
        assert(switch_log_pi.shape[1] == 21)  # num_switches + 1

        # probability sum to 1
        assert(torch.all(torch.abs(torch.sum(
            masked_pi, dim=1) - 1) < 1e-6).item())

        opt = torch.optim.Adam(self.mgcn_policy.parameters(), lr=1e-3)

        # sample actions based on the probability
        switch_p = Categorical(masked_pi)
        switch_a = switch_p.sample()
        switch_a = switch_a.reshape([-1, 1])

        # test policy gradient can flow through
        log_pi_acts = switch_log_pi.gather(1, switch_a)

        # dummy advantage
        adv = torch.FloatTensor(np.random.rand(32, 1))

        # entropy
        entropy = (switch_log_pi * switch_pi).sum(dim=-1).mean()

        # policy gradient
        pg_loss = - (log_pi_acts * adv).mean()

        # loss
        loss = pg_loss + self.entropy_factor * entropy

        # gradient descent
        opt.zero_grad()
        loss.backward()
        opt.step()
