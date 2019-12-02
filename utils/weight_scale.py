import numpy as np
# from param import config


eps = 1e-6
'''
get the mean and max nerual network parameter scale
'''

def get_param_scale(nn):
    total_param = 0
    total_val = 0
    max_val = -np.inf
    state_dict = nn.state_dict()
    for k in state_dict:
        param = state_dict[k]
        p = 1
        for s in param.shape:
            p *= s
        total_param += p
        total_val += param.sum()
        tmp_max = param.max()
        if tmp_max > max_val:
            max_val = tmp_max
    mean_val = total_val / (total_param + eps)

    return mean_val, max_val
