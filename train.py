import numpy as np
import torch
import torch.nn.functional as F
import layers
import msg
import bipartite_gcn
from environment import bipartite_graph

def train(bi_graph):
    u_feats = bi_graph.u_feats()
    v_feats = bi_graph.v_feats()
    n_hids = [5, 5]
    num_rounds = 5
    msg_pass_dirs = ['uu', 'uv', 'vu', 'uv', 'vu']
    bi_gcn = bipartite_gcn.BipartiteGCN(u_feats, v_feats, msg_pass_dirs, n_hids, num_rounds)
    bi_gcn.train()
    optimizer = torch.optim.Adam(bi_gcn.parameters(), lr=1e-2)

    # print([n for n in bi_gcn.parameters()])

    n_iters = 5000
    losses = []
    for _ in range(n_iters):
        u_node_feats_np = bi_graph.u_node_feats()
        v_node_feats_np = bi_graph.v_node_feats()
        uv_adj_mat_np, vu_adj_mat_np = bi_graph.bi_adj_mat()
        true_output_np = bi_graph.true_output()
        print(true_output_np)

        u_node_feats = torch.FloatTensor(u_node_feats_np)
        v_node_feats = torch.FloatTensor(v_node_feats_np)
        uv_adj_mat = torch.FloatTensor(uv_adj_mat_np)
        vu_adj_mat = torch.FloatTensor(vu_adj_mat_np)
        true_output = torch.FloatTensor(true_output_np)

        est_outputs = bi_gcn(u_node_feats, v_node_feats, uv_adj_mat, vu_adj_mat)
        est_outputs = est_outputs.squeeze()
        print(est_outputs)

        loss = torch.sum((true_output - est_outputs) ** 2)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    bi_graph = bipartite_graph.BipartiteGraph(20, 4)
    train(bi_graph)