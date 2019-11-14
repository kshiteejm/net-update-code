import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class FullyConnectNN(nn.Module):
    def __init__(self, in_size, n_hids, out_size, act=nn.LeakyReLU,
                 layer_norm=False, final_layer_act=True):
        '''
        in_size: number of input features
        n_hids: number of hidden neurons (a list)
        out_size: number of output features
        '''
        super(FullyConnectNN, self).__init__()

        self.in_size = in_size
        self.n_hids = n_hids
        self.act = act
        self.out_size = out_size
        self.layer_norm = layer_norm
        self.final_layer_act = final_layer_act

        # parameter dimensions
        layers = [self.in_size]
        layers.extend(self.n_hids)
        layers.append(self.out_size)

        # initialize layer operations
        self.modules = []
        for l in range(len(layers) - 1):
            self.modules.append(nn.Linear(layers[l], layers[l + 1]))
            if l < len(layers) - 2 or final_layer_act:
                self.modules.append(self.act())
            if self.layer_norm:
                self.modules.append(nn.LayerNorm(layers[l + 1]))

        # for forward function
        self.sequential = nn.Sequential(*self.modules)

    def forward(self, in_vec):
        return self.sequential(in_vec)
