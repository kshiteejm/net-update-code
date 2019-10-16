import numpy as np
import torch
import torch.nn.functional as F
import layers
import msg
import bipartite_gcn
from environment import bipartite_graph
import sys
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def train(bi_graph, rounds):
    u_feats = bi_graph.u_feats()
    v_feats = bi_graph.v_feats()
    n_hids = [5, 5]
    num_rounds = rounds
    msg_pass_dirs = ['uu', 'uv', 'vu', 'uv', 'vu']
    bi_gcn = bipartite_gcn.BipartiteGCN(u_feats, v_feats, msg_pass_dirs, n_hids, num_rounds)
    bi_gcn.train()
    optimizer = torch.optim.Adam(bi_gcn.parameters(), lr=1e-2)
    writer = SummaryWriter()

    # print([n for n in bi_gcn.parameters()])

    n_iters = 5000
    n_iter = 0
    losses = []
    last_loss = sys.maxsize

    u_node_feats_np = bi_graph.u_node_feats()
    v_node_feats_np = bi_graph.v_node_feats()
    uv_adj_mat_np, vu_adj_mat_np = bi_graph.bi_adj_mat()
    true_output_np = bi_graph.true_output()

    u_node_feats = torch.FloatTensor(u_node_feats_np)
    v_node_feats = torch.FloatTensor(v_node_feats_np)
    uv_adj_mat = torch.FloatTensor(uv_adj_mat_np)
    vu_adj_mat = torch.FloatTensor(vu_adj_mat_np)
    true_output = torch.FloatTensor(true_output_np)

    # print("==Inputs==")
    # print(u_node_feats)
    # print(v_node_feats)
    # print(uv_adj_mat)
    # print(vu_adj_mat)
    # print("==========")

    while last_loss > 20:
        est_output = bi_gcn(u_node_feats, v_node_feats, uv_adj_mat, vu_adj_mat)
        est_output = est_output.squeeze()

        loss = torch.sum((true_output - est_output) ** 2)
        last_loss = loss.data

        if n_iter%1000 == 0:
            print("n_iter: %d" % n_iter)
            print(true_output)
            print(est_output)
            print("Loss/Train: %s" % last_loss)
        
        writer.add_scalar('Loss/Train, Rounds = %d' % num_rounds, last_loss, n_iter)

        # losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter = n_iter + 1

    print("n_iter: %d" % n_iter)
    print(true_output)
    print(est_output)
    print("Loss/Train: %s" % last_loss)

    return n_iter

if __name__ == '__main__':
    num_links = 20
    num_flows = 4
    num_instances = 20
    range_rounds = [1, 2, 3, 4, 5]
    range_n_iters = np.zeros((num_instances, len(range_rounds)))
    normalized_n_iters = np.zeros((num_instances, len(range_rounds)))
    for instance in range(num_instances): 
        random.seed(instance)
        bi_graph = bipartite_graph.BipartiteGraph(num_links, num_flows)
        for rounds in range_rounds:
            n_iters = train(bi_graph, rounds)
            range_n_iters[instance][rounds - 1] = n_iters
    for row_index in range(len(range_n_iters)):
        min_iters = min(range_n_iters[row_index, :])
        for col_index in range(len(range_n_iters[row_index, :])):
            normalized_n_iters[row_index][col_index] = min_iters/range_n_iters[row_index][col_index]
    
    palette = plt.get_cmap("tab20")
    fig = plt.figure()
    for row_index in range(len(range_n_iters)):
        plt.plot(range_rounds, normalized_n_iters[row_index,:], marker='', color=palette(row_index), linewidth=1, alpha=0.9)
    plt.title("Normalized Convergence Score for 20 instances")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Normalized Convergence Score (higher is better)")
    plt.savefig("memorization_rounds_%d.png" % num_instances)
