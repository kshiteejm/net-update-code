import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import msg
import layers

class BipartiteGCN(nn.Module):
    def __init__(self, u_feats, v_feats, msg_pass_dirs, n_hids, num_rounds, 
                 act=F.leaky_relu, layer_norm_on=False, final_layer_act=True):
        '''
        Bipartite Graph = (u, v, e)
        u_feats: feature size of each node in U
        v_feats: feature size of each node in V
        n_hids: number of hidden neurons as a list

        num_rounds: number of rounds of message passing
        msg_pass_dirs: message passing directions 
            'uu': transformation on nodes in U (no message passing)
            'vv': transformation on nodes in V (no message passing)
            'uv': message passing + transformation from nodes in U to V
            'vu': message passing + tranformation from nodes in V to U
        '''

        super(BipartiteGCN, self).__init__()

        self.num_rounds = num_rounds
        self.msg_pass_dirs = msg_pass_dirs

        self.f_msg = []
        for msg_pass_dir in msg_pass_dirs:
            if msg_pass_dir == 'uu':
                f = layers.Transformation(
                    u_feats, n_hids, u_feats, act, layer_norm_on, final_layer_act)
                self.f_msg.append(f)
            if msg_pass_dir == 'vv':
                f = layers.Transformation(
                    v_feats, n_hids, v_feats, act, layer_norm_on, final_layer_act)
                self.f_msg.append(f)
            if msg_pass_dir == 'uv':
                f = layers.Transformation(
                    u_feats, n_hids, v_feats, act, layer_norm_on, final_layer_act)
                _msg = msg.OneWayMessagePassing(f)
                self.f_msg.append(_msg)
            if msg_pass_dir == 'vu':
                f = layers.Transformation(
                    v_feats, n_hids, u_feats, act, layer_norm_on, final_layer_act)
                _msg = msg.OneWayMessagePassing(f)
                self.f_msg.append(_msg)

        # self.theta_1 = layers.Transformation(
        #     in_feats=5, n_hids=[5, 5], out_feats=5, 
        #     act=F.leaky_relu, layer_norm_on=False, final_layer_act=True)
        
        # self.theta_2 = layers.Transformation(5, [5, 5], 2)
        # self.msg = msg.OneWayMessagePassing(self.theta_2)

        # self.theta_3 = layers.Transformation(2, [5, 5], 5)
        # self.msg = msg.OneWayMessagePassing(self.theta_3)

        # self.theta_4 = layers.Transformation(2, [5, 5], 5)
        # self.msg = msg.OneWayMessagePassing(self.theta_4)

        # self.theta_5 = layers.Transformation(5, [5, 5], 2)
        # self.msg = msg.OneWayMessagePassing(self.theta_5)

        # self.theta_6 = layers.Transformation(2, [5, 5], 5)
        # self.msg = msg.OneWayMessagePassing(self.theta_6)
        
    def forward(self, u_node_feats, v_node_feats, uv_adj_mat, vu_adj_mat):
        u = u_node_feats
        v = v_node_feats
        msg_pass_dirs = self.msg_pass_dirs
        f_msg = self.f_msg
        for _ in range(self.num_rounds):
            for i in range(len(msg_pass_dirs)):
                if msg_pass_dirs[i] == 'uu':
                    u = f_msg[i](u)
                if msg_pass_dirs[i] == 'vv':
                    v = f_msg[i](v)
                if msg_pass_dirs[i] == 'uv':
                    v = f_msg[i](u, uv_adj_mat)
                if msg_pass_dirs[i] == 'vu':
                    u = f_msg[i](v, vu_adj_mat)
        
        return v_node_feats
