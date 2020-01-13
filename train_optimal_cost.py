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
from utils.weight_scale import get_param_scale
import sys


def train(seed, dataset, dataset_size, model_dir):
    pods = 4

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

    if dataset_size > 0:
        nodefeats_file_list = nodefeats_file_list[:dataset_size]
        adjmats_file_list = adjmats_file_list[:dataset_size]
        cost_file_list = cost_file_list[:dataset_size]

    nodefeats_rows = []
    for f in nodefeats_file_list:
        f = "%s/%s" % (dataset, f)
        row = np.load(f)
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
    num_epochs = 401
    training_index_limit = int(len(cost_file_list) * 0.8 / 32) * 32
    training_indexes = list(range(0, training_index_limit))
    validation_indexes = list(range(training_index_limit, len(cost_file_list)))

    print('Setting up monitoring...')
    monitor = SummaryWriter('./results/' +
        time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))

    n_iter = 0
    proj_done_time = ProjectFinishTime(num_epochs, same_line=False)
    for n_epoch in range(num_epochs):
        if n_epoch%10 == 0: 
            torch.save(mgcn_value.state_dict(), 
                       "%s/model_trained_%s_epoch.pt" % (model_dir, n_epoch))
        
        random.shuffle(training_indexes)
        opt.zero_grad()
        batch_loss = 0.0
        validation_loss = 0.0

        for i in range(len(training_indexes)):
            if i%32 == 0 or i == (len(training_indexes) - 1):
                if i == (len(training_indexes) - 1): 
                    index = training_indexes[i]
                    for type_i in range(num_types):
                        batch_node_feats[type_i].append(nodefeats_rows[index][type_i])
                        batch_adj_mats[type_i].append(adjmats_rows[index][type_i])
                    batch_cost_target.append(cost_rows[index])
                
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
                    
                    accuracy = ((batch_cost_estimate - cost_target_torch) \
                               / (cost_target_torch + 1e-6)).mean()
                    monitor.add_scalar('Loss/train_accuracy', accuracy.item(), n_iter)

                    maxima = max(abs(batch_cost_estimate - cost_target_torch))
                    monitor.add_scalar('Loss/train_max_diff', maxima.item(), n_iter)

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

        param_mean, param_max = get_param_scale(mgcn_value)
        monitor.add_scalar('Parameters/parameter_mean', param_mean, n_epoch)
        monitor.add_scalar('Parameters/parameter_max', param_max, n_epoch)

        monitor.add_scalar('Loss/validation_loss', validation_loss, n_epoch)

        accuracy = ((batch_cost_estimate - cost_target_torch) \
                   / (cost_target_torch + 1e-6)).mean()
        monitor.add_scalar('Loss/validation_accuracy', accuracy.item(), n_epoch)

        maxima = max(abs(batch_cost_estimate - cost_target_torch))
        monitor.add_scalar('Loss/validation_max_diff', maxima.item(), n_epoch)

        proj_done_time.update_progress(n_epoch, message="validation")

def test(seed, dataset, dataset_size, model_dir, n_epoch, test_size, test_steps_left): 
    pods = 4
    num_switches = (pods//2) * (pods//2 + 2*pods) # assuming fat-tree
    max_num_steps = 4

    file_list = os.listdir(dataset)
    
    if test_steps_left > 0:
        substring = "nodefeats_fat_tree_%s_pods_%s_%s" % (pods, seed, test_steps_left)
        nodefeats_file_list = [item for item in file_list if substring in item]
        nodefeats_file_list.sort()

        substring = "adjmats_fat_tree_%s_pods_%s_%s" % (pods, seed, test_steps_left)
        adjmats_file_list = [item for item in file_list if substring in item]
        adjmats_file_list.sort()

        substring = "cost_fat_tree_%s_pods_%s_%s" % (pods, seed, test_steps_left)
        cost_file_list = [item for item in file_list if substring in item]
        cost_file_list.sort()
    else:
        substring = "nodefeats_fat_tree_%s_pods_%s" % (pods, seed)
        nodefeats_file_list = [item for item in file_list if substring in item]
        nodefeats_file_list.sort()

        substring = "adjmats_fat_tree_%s_pods_%s" % (pods, seed)
        adjmats_file_list = [item for item in file_list if substring in item]
        adjmats_file_list.sort()

        substring = "cost_fat_tree_%s_pods_%s" % (pods, seed)
        cost_file_list = [item for item in file_list if substring in item]
        cost_file_list.sort()

    if test_size > 0:
        nodefeats_file_list = nodefeats_file_list[:test_size]
        adjmats_file_list = adjmats_file_list[:test_size]
        cost_file_list = cost_file_list[:test_size]

    nodefeats_rows = []
    for f in nodefeats_file_list:
        f = "%s/%s" % (dataset, f)
        row = np.load(f)
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

    mgcn_value = Batch_MGCN_Value(
                        n_switches=20,
                        n_feats=[2, 2, 2, 2],
                        n_output=8,
                        n_hids=[16, 32],
                        h_size=8,
                        n_steps=8, 
                        layer_norm_on=True)

    state_dicts = torch.load("%s/model_trained_%s_epoch.pt" % (model_dir, n_epoch))
    mgcn_value.load_state_dict(state_dicts)
    print("LOADED SUCCESSFULLY.")

    num_types = len(nodefeats_rows[0])
    training_index_limit = int(len(cost_file_list) * 0.8 / 32) * 32
    training_indexes = list(range(0, training_index_limit))
    validation_indexes = list(range(training_index_limit, len(cost_file_list)))

    batch_update_switch_set_strings = []
    batch_num_steps_left = []
    batch_cost_target_list = []
    batch_cost_estimate_list = []
    batch_l2_loss = []
    
    proj_done_time = ProjectFinishTime(len(cost_file_list), same_line=False)

    for i in range(len(cost_file_list)):

        batch_node_feats = []
        batch_adj_mats = []
        batch_cost_target = []
        for _ in range(num_types):
            batch_node_feats.append([])
            batch_adj_mats.append([])
        
        index = i
        for type_i in range(num_types):
            batch_node_feats[type_i].append(nodefeats_rows[index][type_i])
            batch_adj_mats[type_i].append(adjmats_rows[index][type_i])
            if type_i == 0:
                num_steps_left = \
                    int(nodefeats_rows[index][type_i][0][1]*max_num_steps)
                switch_set_string = ""
                for switch_id in range(num_switches):
                    if int(nodefeats_rows[index][type_i][switch_id][0]) == 1:
                        switch_set_string = switch_set_string + str(switch_id) + ","
                switch_set_string = switch_set_string[:-1]
                batch_num_steps_left.append(num_steps_left)
                batch_update_switch_set_strings.append(switch_set_string)

        batch_cost_target.append(cost_rows[index])
        batch_cost_target_list.append(cost_rows[index][0])
        
        node_feats_torch = [torch.FloatTensor(nf) \
                            for nf in batch_node_feats]
        adj_mats_torch  = [torch.FloatTensor(adj) \
                            for adj in batch_adj_mats]
        cost_target_torch = torch.FloatTensor(batch_cost_target)

        batch_cost_estimate = mgcn_value(node_feats_torch, adj_mats_torch)
        batch_cost_estimate_list.append(batch_cost_estimate.detach().numpy()[0][0])

        # l2 loss
        l2_loss = torch.nn.MSELoss(reduction='mean')
        loss = l2_loss(batch_cost_estimate, cost_target_torch)
        validation_loss = loss.data.item()
        batch_l2_loss.append(validation_loss)

        proj_done_time.update_progress(i, "elapsed")

    f_estimate = open("values_model_estimated.csv", 'w')
    f_target = open("values_target.csv", "w")
    batch_cost_estimate_numpy = batch_cost_estimate.detach().numpy()
    for i in range(len(batch_update_switch_set_strings)):
        cost_target = batch_cost_target_list[i]
        cost_estimate = batch_cost_estimate_list[i]
        num_steps_left = batch_num_steps_left[i]
        update_switch_set_string = batch_update_switch_set_strings[i]
        f_estimate.write("%s,%s,%s\n" % 
                         (cost_estimate, num_steps_left, update_switch_set_string))
        f_target.write("%s,%s,%s\n" % 
                       (cost_target, num_steps_left, update_switch_set_string))
    f_estimate.close()
    f_target.close()

    validation_loss = sum(batch_l2_loss)/len(batch_l2_loss)

    print('l2 loss: {}'.format(validation_loss))

    plot_size = test_size
    if test_size <= 0:
        plot_size = 200
    
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(batch_cost_target_list[:plot_size])
    plt.plot(batch_cost_estimate_list[:plot_size])
    plt.legend(['target', 'estimate'])
    plt.title('l2 loss: {}'.format(validation_loss))
    plt.savefig('./epoch_{}_{}_{}.png'.format(n_epoch, test_steps_left, seed))


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("python3 train_optimal_cost.py \
                       seed \
                       train_bool \
                       dataset_loc \
                       dataset_size \
                       model_dir \
                       n_epoch \
                       test_size \
                       test_steps_left")
    seed = sys.argv[1]
    random.seed(seed)
    is_train = ("True" == sys.argv[2])
    dataset = sys.argv[3]
    dataset_size = int(sys.argv[4])
    model_dir = sys.argv[5]
    if is_train:
        train(seed, dataset, dataset_size, model_dir)
    else:
        n_epoch = int(sys.argv[6])
        test_size = 0
        test_steps_left = 0
        if len(sys.argv) >= 8:
            test_size = int(sys.argv[7])
        if len(sys.argv) >= 9:
            test_steps_left = int(sys.argv[8])
        test(seed, dataset, dataset_size, model_dir, n_epoch, test_size, test_steps_left)
