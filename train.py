import time
import torch
import numpy as np
from param import config
from learn.v import value_train
from learn.pg import policy_gradient
from environment.rl_interface import RLEnv
from torch.distributions import Categorical
from utils.gen_traj import TrajectoryGenerator
from gcn.batch_mgcn_value import Batch_MGCN_Value
from gcn.batch_mgcn_policy import Batch_MGCN_Policy
from utils.state_transform import get_tensors
from utils.rewards import cumulative_rewards, gae_advantage


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
        (config.batch_size, node_feat.shape[0],
        node_feat.shape[1]), dtype=np.float32) for \
        node_feat in node_feats]
    batch_next_node_feats = [np.zeros(
        (config.batch_size, node_feat.shape[0],
        node_feat.shape[1]), dtype=np.float32) for \
        node_feat in node_feats]
    batch_adj_mats = [np.zeros(
        (config.batch_size, adj_mat.shape[0],
        adj_mat.shape[1]), dtype=np.float32) for \
        adj_mat in adj_mats]
    batch_switch_masks = np.zeros(
        (config.batch_size, switch_mask.shape[0]),
        dtype=np.float32)
    batch_actions = np.zeros(
        (config.batch_size, 1), dtype=np.int64)  # Long tensor
    batch_rewards =  np.zeros(
        (config.batch_size, 1), dtype=np.float32)
    batch_dones = batch_rewards =  np.zeros(
        (config.batch_size, 1), dtype=np.float32)

    # perform training
    for train_iter in range(config.num_epochs):

        for ba in range(config.batch_size):
            node_feats_torch, adj_mats_torch, switch_mask_torch = \
                get_tensors(node_feats, adj_mats, switch_mask)

            # feed forward policy net
            switch_log_pi, switch_pi, masked_pi = policy_net(
                node_feats_torch, adj_mats_torch, switch_mask_torch)

            # sample action
            switch_p = Categorical(masked_pi)
            switch_a = switch_p.sample().item()

            next_node_feats, next_adj_mats, \
                next_switch_mask, reward, done = traj_gen.step(switch_a)

            # store into storage
            for i in range(len(batch_node_feats)):
                batch_node_feats[i][ba, :, :] = node_feats[i][:, :]
                batch_next_node_feats[i][ba, :, :] = next_node_feats[i][:, :]
                batch_adj_mats[i][ba, :, :] = adj_mats[i][:, :]
            batch_switch_masks[ba, :] = switch_mask
            batch_actions[ba, :] = switch_a
            batch_rewards[ba, :] = reward
            batch_dones[ba, :] = done

            # state advance to next step
            node_feats = next_node_feats
            adj_mats = next_adj_mats
            switch_mask = next_switch_mask

        # torchify everything
        batch_node_feats_torch = torch.from_numpy(batch_node_feats)
        batch_next_node_feats_torch = torch.from_numpy(batch_next_node_feats)
        batch_adj_mats_torch = torch.from_numpy(batch_adj_mats)
        batch_switch_masks_torch = torch.from_numpy(batch_switch_masks)
        batch_actions_torch = torch.from_numpy(batch_actions)
        batch_rewards_torch = torch.from_numpy(batch_rewards)
        batch_dones_torch = torch.from_numpy(batch_dones)

        # compute values
        values_with_grad = value_net(batch_node_feats_torch, adj_mats_torch)
        values_np = values_with_grad.detach().numpy()
        next_values_np = value_net(batch_next_node_feats_torch,
                                batch_adj_mats_torch).detach().numpy()

        # aggregate reward
        returns_np = cumulative_rewards(
            rewards, dones, config.gamma, next_values_np)
        returns = torch.from_numpy(returns_np)

        # policy gradient
        adv = gae_advantage(rewards, dones, values_np,
            next_values_np, config.gamma, config.lam,
            config.adv_norm)
        adv = torch.from_numpy(adv)

        # value gradient
        pg_loss, entropy = policy_gradient(
            policy_net, policy_opt,
            batch_node_feats_torch, batch_adj_mats_torch, batch_adj_mats_torch,
            batch_actions, adv, entropy_factor)

        # value training
        v_loss = value_train(value_net,
            values_with_grad, returns)

        # update entropy factor


        # monitor

        # save model, do testing



if __name__ == '__main__':
    main()
