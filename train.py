import time
import torch
from param import config
from environment.rl_interface import RLEnv
from torch.distributions import Categorical
from utils.gen_traj import TrajectoryGenerator
from gcn.batch_mgcn_value import Batch_MGCN_Value
from gcn.batch_mgcn_policy import Batch_MGCN_Policy


def main():

    # policy network for taking actions and policy gradient
    policy_net = Batch_MGCN_Policy(
        config.num_switches, [config.class_feat, config.path_feat,
        config.link_feat, config.switch_feat], config.n_output,
        config.hid_dim, config.h_size, config.n_steps)

    # value network for value prediction and advantage estimation
    value_net = Batch_MGCN_Value(
        config.num_switches, [config.class_feat, config.path_feat,
        config.link_feat, config.switch_feat], config.n_output,
        config.hid_dim, config.h_size, config.n_steps)

    # trajectory generator
    traj_gen = TrajectoryGenerator()

    # initial state
    node_feats, adj_mats, switch_mask = traj_gen.reset()

    # storage for batch training
    batch_node_feats = [np.zeros(
        config.batch_size, node_feat.shape[1],
        node_feat.shape[2], dtype=np.float32) for \
        node_feat in node_feats]
    batch_next_node_feats = []
    batch_adj_mats =  # Float tensor
    batch_switch_masks = 
    batch_actions =   # Long tensor
    batch_rewards =  
    batch_dones =  # Float tensor 

    # perform training
    for train_iter in range(config.num_epochs):
        for ba in range(config.batch_size):

            # feed forward policy net
            switch_log_pi, switch_pi, masked_pi = policy_net(
                node_feats_torch, adj_mats_torch, switch_mask)

            # sample action
            switch_p = Categorical(masked_pi)
            switch_a = switch_p.sample()
            switch_a = switch_a.reshape([-1, 1])

            next_node_feats_torch, next_adj_mats_torch, \
                next_switch_mask_torch, reward, done = traj_gen.step(switch_a)

            # store into storage
            batch_node_feats[] = node_feats_torch
            batch_next_node_feats[] = next_node_feats_torch
            batch_adj_mats[] = adj_mats_torch
            batch_switch_masks[] = switch_mask_torch
            batch_actions[] = switch_a
            batch_rewards[] = reward
            batch_dones[] = done

            # state advance to next step
            node_feats_torch = next_node_feats_torch
            adj_mats_torch = next_adj_mats_torch
            switch_mask = next_switch_mask_torch

        # compute values

        # aggregate reward
        cum_rewards = cumulative_rewards(
            rewards, dones, config.gamma, next_values_np)

        # policy gradient

        # value gradient

        # update entropy factor

        # monitor

        # save model, do testing



if __name__ == '__main__':
    main()
