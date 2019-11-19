import os
import time
import torch
import torch.nn as nn
import numpy as np
from gcn.mgcn_value import MGCN_Value
from gcn.layers import FullyConnectNN
import random
from torch.utils.tensorboard import SummaryWriter


def train():
    pods = 4
    num_steps = 4
    max_link_bw = 10000

    # read dataset
    dataset = "/data/kshiteej/net-update-data"
    file_list = os.listdir(dataset)
    
    substring = "nodefeats_fat_tree_%s_pods_" % pods
    nodefeats_file_list = [item for item in file_list if substring in item]
    nodefeats_file_list.sort()

    substring = "adjmats_fat_tree_%s_pods_" % pods
    adjmats_file_list = [item for item in file_list if substring in item]
    adjmats_file_list.sort()

    substring = "cost_fat_tree_%s_pods_" % pods
    cost_file_list = [item for item in file_list if substring in item]
    cost_file_list.sort()

    nodefeats_rows = []
    for f in nodefeats_file_list:
        f = "%s/%s" % (dataset, f)
        row = np.load(f)
        # normalize
        row[0][:,1] /= num_steps
        row[1] /= max_link_bw
        row[3] /= max_link_bw
        nodefeats_rows.append(
            [torch.FloatTensor(nf) \
            for nf in row])
    print("Finished Reading Node Features...")

    adjmats_rows = []
    for f in adjmats_file_list:
        f = "%s/%s" % (dataset, f)
        adjmats_rows.append(
            [torch.FloatTensor(nf) \
            for nf in np.load(f)])
    print("Finished Reading Adjacency Matrices...")

    cost_rows = []
    for f in cost_file_list:
        f = "%s/%s" % (dataset, f)
        cost_rows.append(
            torch.FloatTensor(np.load(f)))
    print("Finished Reading Optimal Costs...")
    
    assert(len(nodefeats_rows) == len(adjmats_rows))
    assert(len(adjmats_rows) == len(cost_rows))

    # setup training
    l2_loss = torch.nn.MSELoss(reduction='mean')
    
    mgcn_value = MGCN_Value(
                        n_switches=20,
                        n_feats=[2, 2, 2, 2],
                        n_output=8,
                        n_hids=[16, 32],
                        h_size=8,
                        n_steps=8, 
                        layer_norm_on=False)

    opt = torch.optim.Adam(mgcn_value.parameters(), lr=1e-3)

    num_epochs = 10000
    training_index_limit = int(len(cost_file_list) * 0.8 / 32) * 32
    training_indexes = list(range(0, training_index_limit))
    validation_indexes = list(range(training_index_limit, len(cost_file_list)))

    print('Setting up monitoring...')
    monitor = SummaryWriter('./results/' +
        time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))

    # # normalize the input features
    # # TODO: normalize it before converting to torch tensor
    # for i in range(len(nodefeats_rows)):
    #     nodefeats_rows[i][0] /= 4
    #     nodefeats_rows[i][1] /= 10000
    #     nodefeats_rows[i][3] /= 10000

    n_iter = 0
    for n_epoch in range(num_epochs):
        random.shuffle(training_indexes)
        opt.zero_grad()
        batch_loss = 0.0
        validation_loss = 0.0
        
        for i in range(len(training_indexes)):
            if i > 0 and i%32 == 0:
                monitor.add_scalar('Loss/train_loss', batch_loss, n_iter)
                loss = 0.0
                opt.step()
                opt.zero_grad()
            
            index = training_indexes[i]

            node_feats_torch = nodefeats_rows[index]
            adj_mats_torch  = adjmats_rows[index]
            cost_target = cost_rows[index]

            cost_estimate = mgcn_value(
                node_feats_torch, adj_mats_torch)

            # l2 loss
            loss = l2_loss(cost_estimate, cost_target)
            batch_loss = loss.data.item() + batch_loss

            # backward
            loss.backward()
            n_iter = n_iter + 1

        for i in range(len(validation_indexes)):
            index = validation_indexes[i]
            node_feats_torch = nodefeats_rows[index]
            adj_mats_torch  = adjmats_rows[index]
            cost_target = cost_rows[index]

            cost_estimate = mgcn_value(
                node_feats_torch, adj_mats_torch)

            # l2 loss
            loss = l2_loss(cost_estimate, cost_target)
            validation_loss = loss.data.item() + validation_loss
        
        monitor.add_scalar('Loss/validation_loss', validation_loss, n_iter)

    torch.save(mgcn_value.state_dict(), "model.pt")

if __name__ == '__main__':
    random.seed(42)
    train()
