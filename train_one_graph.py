import os
import time
import torch
import torch.nn as nn
import numpy as np
from gcn.mgcn_value import MGCN_Value
from gcn.layers import FullyConnectNN
from torch.utils.tensorboard import SummaryWriter


def train():
    seed = 1
    pods = 4

    dataset = "./data"
    
    substring = "nodefeats_fat_tree_%s_pods_%s" % (pods, seed)
    nodefeats_file_list =  os.listdir("./data")
    nodefeats_file_list = [item for item in nodefeats_file_list if substring in item]

    substring = "adjmats_fat_tree_%s_pods_%s" % (pods, seed)
    adjmats_file_list =  os.listdir("./data")
    adjmats_file_list = [item for item in adjmats_file_list if substring in item]

    substring = "cost_fat_tree_%s_pods_%s" % (pods, seed)
    cost_file_list =  os.listdir("./data")
    cost_file_list = [item for item in cost_file_list if substring in item]
    
    nodefeats_rows = []
    for f in nodefeats_file_list:
        f = "%s/%s" % (dataset, f)
        nodefeats_rows.append(
            [torch.FloatTensor(nf) \
            for nf in np.load(f)])

    adjmats_rows = []
    for f in adjmats_file_list:
        f = "%s/%s" % (dataset, f)
        adjmats_rows.append(
            [torch.FloatTensor(nf) \
            for nf in np.load(f)])

    cost_rows = []
    for f in cost_file_list:
        f = "%s/%s" % (dataset, f)
        cost_rows.append(
            # [torch.FloatTensor(nf) \
            # for nf in np.load(f)])
            torch.FloatTensor([0]))

    assert(len(nodefeats_rows) == len(adjmats_rows))
    assert(len(adjmats_rows) == len(cost_rows))

    l2_loss = torch.nn.MSELoss(reduction='mean')
    
    mgcn_value = MGCN_Value(
                        n_switches=20,
                        n_feats=[2, 2, 2, 2],
                        n_output=8,
                        n_hids=[16, 32],
                        h_size=8,
                        n_steps=8, 
                        layer_norm_on = False)

    opt = torch.optim.Adam(mgcn_value.parameters(), lr=1e-3)

    num_training_iterations = 10000

    print('Setting up monitoring..')
    monitor = SummaryWriter('./results/' +
        time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))

    for n_iter in range(num_training_iterations):

        opt.zero_grad()
        agg_loss = []
        for i in range(len(nodefeats_rows)):
            node_feats_torch = nodefeats_rows[i]
            adj_mats_torch  = adjmats_rows[i]
            cost_target = cost_rows[i]

            cost_estimate = mgcn_value(
                node_feats_torch, adj_mats_torch)

            # l2 loss
            loss = l2_loss(cost_estimate, cost_target)
            agg_loss.append(loss.data)

            # backward
            loss.backward()
        
        opt.step()

        monitor.add_scalar('Loss/loss_1', agg_loss[0].item(), n_iter)
        monitor.add_scalar('Loss/loss_2', agg_loss[1].item(), n_iter)
        monitor.add_scalar('Loss/loss_3', agg_loss[2].item(), n_iter)

        if n_iter % 100 == 0:
            print("n_iter: %d" % n_iter)
            print("loss: %s" % agg_loss)

    torch.save(mgcn_value.state_dict(), "model.pt")

if __name__ == '__main__':
    train()
