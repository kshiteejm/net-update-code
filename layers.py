# source from https://github.com/hongzimao/gcn_pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Transformation(nn.Module):
    def __init__(self, in_feats, n_hids, out_feats, act=F.leaky_relu,
                 layer_norm_on=False, final_layer_act=True):
        '''
        in_feats: number of input features
        n_hids: number of hidden neurons (a list)
        out_feats: number of output features
        '''
        super(Transformation, self).__init__()

        self.in_feats = in_feats
        self.n_hids = n_hids
        self.out_feats = out_feats
        self.layer_norm_on = layer_norm_on
        self.final_layer_act = final_layer_act

        # parameter dimensions
        layers = [self.in_feats]
        layers.extend(self.n_hids)
        layers.append(self.out_feats)

        self.weights = []
        self.biases = []
        self.layer_norms = []
        for l in range(len(layers) - 1):
            self.weights.append(Parameter(torch.FloatTensor(
                layers[l], layers[l + 1])))
            self.biases.append(Parameter(torch.FloatTensor(
                layers[l + 1])))
            if self.layer_norm_on:
                self.layer_norms.append(nn.LayerNorm(layers[l + 1]))

        self.weights = nn.ParameterList(self.weights)
        self.biases = nn.ParameterList(self.biases)

        # relu activation function
        self.act = act

        # initialize parameters
        self.reset()

    def reset(self):
        # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
        for weight in self.weights:
            init_range = np.sqrt(6 / (weight.shape[0] + weight.shape[1]))
            weight.data.uniform_(-init_range, init_range)
        for bias in self.biases:
            bias.data.zero_()

    def forward(self, in_vec):
        x = in_vec
        for l in range(len(self.weights)):
            x = torch.mm(x, self.weights[l])
            x = x + self.biases[l]
            if l < len(self.weights) - 1 or self.final_layer_act:
                x = self.act(x)
            if self.layer_norm_on:
                x = self.layer_norms[l](x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_feats) + ' -> ' \
               + ' -> '.join(str(i) for i in self.n_hids) \
               + ' -> ' + str(self.out_feats) + ')'