import os
import time
from proj_time import ProjectFinishTime
import torch
import torch.nn as nn
import numpy as np
from gcn.mgcn_value import MGCN_Value
from gcn.batch_mgcn_value import Batch_MGCN_Value
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
        nodefeats_rows.append(row)
    print("Finished Reading Node Features...")

    adjmats_rows = []
    for f in adjmats_file_list:
        f = "%s/%s" % (dataset, f)
        row = np.load(f)
        adjmats_rows.append(row)
    print("Finished Reading Adjacency Matrices...")

    cost_rows = []
    for f in cost_file_list:
        f = "%s/%s" % (dataset, f)
        row = np.load(f)
        cost_rows.append(row)
    print("Finished Reading Optimal Costs...")
    
    assert(len(nodefeats_rows) == len(adjmats_rows))
    assert(len(adjmats_rows) == len(cost_rows))

    # setup training
    l2_loss = torch.nn.MSELoss(reduction='mean')
    
    mgcn_value = Batch_MGCN_Value(
                        n_switches=20,
                        n_feats=[2, 2, 2, 2],
                        n_output=8,
                        n_hids=[16, 32],
                        h_size=8,
                        n_steps=8, 
                        layer_norm_on=True)

    opt = torch.optim.Adam(mgcn_value.parameters(), lr=1e-3)

    num_types = len(nodefeats_rows[0])
    num_epochs = 10000
    training_index_limit = int(len(cost_file_list) * 0.8 / 32) * 32
    training_indexes = list(range(0, training_index_limit))
    validation_indexes = list(range(training_index_limit, len(cost_file_list)))

    print('Setting up monitoring...')
    monitor = SummaryWriter('./results/' +
        time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))

    n_iter = 0
    proj_done_time = ProjectFinishTime(num_epochs, same_line=False)
    for n_epoch in range(num_epochs):
        random.shuffle(training_indexes)
        opt.zero_grad()
        batch_loss = 0.0
        validation_loss = 0.0

        for i in range(len(training_indexes)):
            if i%32 == 0:
                if i > 0:
                    node_feats_torch = [torch.FloatTensor(nf) \
                                        for nf in batch_node_feats]
                    adj_mats_torch = [torch.FloatTensor(adj) \
                                      for adj in batch_adj_mats]
                    cost_target_torch = torch.FloatTensor(batch_cost_target)
                    batch_cost_estimate = mgcn_value(
                        node_feats_torch, adj_mats_torch)

                    # l2 loss
                    loss = l2_loss(batch_cost_estimate, cost_target_torch)

                    batch_loss = loss.data.item()
                    monitor.add_scalar('Loss/train_loss', batch_loss, n_iter)
                    
                    accuracy = ((batch_cost_estimate - cost_target_torch) / cost_target_torch + 10e-6).mean()
                    monitor.add_scalar('Loss/train_accuracy', accuracy.item(), n_iter)

                    # backward
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                batch_node_feats = []
                batch_adj_mats = []
                batch_cost_target = []
                for _ in range(num_types):
                    batch_node_feats.append([])
                    batch_adj_mats.append([])
            
            index = training_indexes[i]
            for type_i in range(num_types):
                batch_node_feats[type_i].append(nodefeats_rows[index][type_i])
                batch_adj_mats[type_i].append(adjmats_rows[index][type_i])
            batch_cost_target.append(cost_rows[index])
            
            n_iter = n_iter + 1
        
        proj_done_time.update_progress(n_epoch, message="single epoch training")

        batch_node_feats = []
        batch_adj_mats = []
        batch_cost_target = []
        for _ in range(num_types):
            batch_node_feats.append([])
            batch_adj_mats.append([])
        for i in range(len(validation_indexes)):
            index = validation_indexes[i]
            for type_i in range(num_types):
                batch_node_feats[type_i].append(nodefeats_rows[index][type_i])
                batch_adj_mats[type_i].append(adjmats_rows[index][type_i])
            batch_cost_target.append(cost_rows[index])
        
        node_feats_torch = [torch.FloatTensor(nf) \
                            for nf in batch_node_feats]
        adj_mats_torch  = [torch.FloatTensor(adj) \
                           for adj in batch_adj_mats]
        cost_target_torch = torch.FloatTensor(batch_cost_target)

        batch_cost_estimate = mgcn_value(
            node_feats_torch, adj_mats_torch)

        # l2 loss
        loss = l2_loss(batch_cost_estimate, cost_target_torch)
        validation_loss = loss.data.item()
        
        monitor.add_scalar('Loss/validation_loss', validation_loss, n_iter)

        accuracy = ((batch_cost_estimate - cost_target_torch) / cost_target_torch + 10e-6).mean()
        monitor.add_scalar('Loss/validation_accuracy', accuracy.item(), n_iter)
        
        proj_done_time.update_progress(n_epoch, message="validation")

    torch.save(mgcn_value.state_dict(), "model_trained.pt")

if __name__ == '__main__':
    random.seed(42)
    train()
