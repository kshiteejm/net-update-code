import sys
import time
import torch
import numpy as np
from param import config
from learn.v import value_train
from learn.pg import policy_gradient
from utils.rewards import gae_advantage
from environment.rl_interface import RLEnv
from torch.distributions import Categorical
from utils.rewards import cumulative_rewards
from utils.proj_time import ProjectFinishTime
from utils.state_transform import get_tensors
from utils.gen_traj import TrajectoryGenerator
from gcn.batch_mgcn_value import Batch_MGCN_Value
from torch.utils.tensorboard import SummaryWriter
from gcn.batch_mgcn_policy import Batch_MGCN_Policy
from utils.rewards import get_monitor_total_rewards


def main():

    # reproducibility
    torch.manual_seed(config.seed)

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
    
    if config.saved_policy_model is not None:
        policy_net.load_state_dict(torch.load(config.saved_policy_model))

    if config.saved_value_model is not None:
        value_net.load_state_dict(torch.load(config.saved_value_model))

    # optimizer
    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=config.lr_rate)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=config.lr_rate)

    # trajectory generator
    traj_gen = TrajectoryGenerator()

    # initial state
    node_feats, adj_mats, switch_mask = traj_gen.reset()

    # storage for batch training
    batch_node_feats = [np.zeros(
        (config.batch_size, node_feat.shape[0],
        node_feat.shape[1]), dtype=np.float32) for node_feat in node_feats]
    batch_next_node_feats = [np.zeros(
        (config.batch_size, node_feat.shape[0],
        node_feat.shape[1]), dtype=np.float32) for node_feat in node_feats]
    batch_adj_mats = [np.zeros(
        (config.batch_size, adj_mat.shape[0],
        adj_mat.shape[1]), dtype=np.float32) for adj_mat in adj_mats]
    batch_next_adj_mats = [np.zeros(
        (config.batch_size, adj_mat.shape[0],
        adj_mat.shape[1]), dtype=np.float32) for adj_mat in adj_mats]
    batch_switch_masks = np.zeros(
        (config.batch_size, switch_mask.shape[0]), dtype=np.float32)
    batch_actions = np.zeros(
        (config.batch_size, 1), dtype=np.int64)  # Long tensor
    batch_rewards =  np.zeros(
        (config.batch_size, 1), dtype=np.float32)
    batch_dones =  np.zeros(
        (config.batch_size, 1), dtype=np.float32)

    # initialize entropy factor
    entropy_factor = config.entropy_factor

    # project finish time
    proj_progress = ProjectFinishTime(config.num_epochs - (config.start_epoch))

    # result monitoring
    monitor = SummaryWriter(config.result_folder +
        time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))

    manual_action_sequence = [4, 8, 12, 16, 0, 1, 20, 5, 9, 13, 17, 2, 3, 20, 20]
    assert(config.batch_size == len(manual_action_sequence))
    pi_action_store = np.zeros(config.batch_size)
    values_store = np.zeros(config.batch_size)
    returns_store = np.zeros(config.batch_size)
    adv_store = np.zeros(config.batch_size)

    # perform training
    for epoch in range(config.start_epoch, config.num_epochs):

        for ba in range(config.batch_size):
            node_feats_torch, adj_mats_torch, switch_mask_torch = \
                get_tensors(node_feats, adj_mats, switch_mask)

            # feed forward policy net
            switch_log_pi, switch_pi, masked_pi = policy_net(
                node_feats_torch, adj_mats_torch, switch_mask_torch)
            # sample action
            switch_p = Categorical(masked_pi)
            switch_a = switch_p.sample().item()

            switch_a = manual_action_sequence[ba]

            print('ba: {}, pi: {}, action: {}, pi_action: {}'.format(ba, masked_pi, switch_a, 
                    masked_pi[0, switch_a]))
            pi_action_store[ba] = masked_pi[0, switch_a].item()

            next_node_feats, next_adj_mats, \
                next_switch_mask, reward, done = traj_gen.step(switch_a)

            # store into storage
            for i in range(len(batch_node_feats)):
                batch_node_feats[i][ba, :, :] = node_feats[i][:, :]
                batch_next_node_feats[i][ba, :, :] = next_node_feats[i][:, :]
                batch_adj_mats[i][ba, :, :] = adj_mats[i][:, :]
                batch_next_adj_mats[i][ba, :, :] = next_adj_mats[i][:, :]
            batch_switch_masks[ba, :] = switch_mask
            batch_actions[ba, :] = switch_a
            batch_rewards[ba, :] = reward
            batch_dones[ba, :] = done

            # state advance to next step
            node_feats = next_node_feats
            adj_mats = next_adj_mats
            switch_mask = next_switch_mask

        # torchify everything
        batch_node_feats_torch = [torch.from_numpy(ft) for ft in batch_node_feats] 
        batch_next_node_feats_torch = [torch.from_numpy(ft) for ft in batch_next_node_feats]
        batch_adj_mats_torch = [torch.from_numpy(ft) for ft in batch_adj_mats]
        batch_next_adj_mats_torch = [torch.from_numpy(ft) for ft in batch_next_adj_mats]
        batch_switch_masks_torch = torch.from_numpy(batch_switch_masks)
        batch_actions_torch = torch.from_numpy(batch_actions)
        batch_rewards_torch = torch.from_numpy(batch_rewards)
        batch_dones_torch = torch.from_numpy(batch_dones)
        batch_states_torch = (batch_node_feats_torch, batch_adj_mats_torch, batch_switch_masks_torch)

        # compute values
        values_with_grad = value_net(batch_node_feats_torch, batch_adj_mats_torch)
        values_np = values_with_grad.detach().numpy()
        next_values_np = value_net(batch_next_node_feats_torch,
                                batch_next_adj_mats_torch).detach().numpy()

        # aggregate reward
        returns_np = cumulative_rewards(
            batch_rewards, batch_dones, config.gamma, next_values_np)
        returns = torch.from_numpy(returns_np)

        # policy gradient
        adv_np = gae_advantage(batch_rewards, batch_dones, values_np,
            next_values_np, config.gamma, config.lam,
            norm=config.adv_norm)
        adv = torch.from_numpy(adv_np)

        values_store[:] = values_np[:, 0]
        returns_store[:] = returns_np[:, 0]
        adv_store[:] = adv_np[:, 0]
        
        # value gradient
        pg_loss, entropy = policy_gradient(
            policy_net, policy_opt,
            batch_states_torch,
            batch_actions_torch, adv, entropy_factor)

        # value training
        v_loss = value_train(value_opt,
            values_with_grad, returns)

        # update entropy factor
        if entropy_factor - config.entropy_factor_decay > config.entropy_factor_min:
            entropy_factor -= config.entropy_factor_decay
        else:
            entropy_factor = config.entropy_factor_min

        # monitor
        proj_progress.update_progress(epoch)
        for i in range(config.batch_size):
            monitor.add_scalar('Policy/action_{}'.format(i), pi_action_store[i], epoch)
            monitor.add_scalar('Value/values_{}'.format(i), values_store[i], epoch)
            monitor.add_scalar('Returns/returns_{}'.format(i), returns_store[i], epoch)
            monitor.add_scalar('Adv/advs_{}'.format(i), adv_store[i], epoch)


if __name__ == '__main__':
    main()
