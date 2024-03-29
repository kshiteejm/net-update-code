import sys
import time
import torch
import numpy as np
from param import config
from learn.v import value_train
from utils.rewards import gae_advantage
from environment.rl_interface import RLEnv
from torch.distributions import Categorical
from utils.rewards import cumulative_rewards
from utils.proj_time import ProjectFinishTime
from utils.state_transform import get_tensors
from utils.gen_traj import TrajectoryGenerator
from learn.pg_no_masking import policy_gradient
from gcn.batch_mgcn_value import Batch_MGCN_Value
from torch.utils.tensorboard import SummaryWriter
from utils.rewards import get_monitor_total_rewards
from gcn.batch_mgcn_policy_no_masking import Batch_MGCN_Policy


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

    # perform training
    for epoch in range(config.start_epoch, config.num_epochs):

        for ba in range(config.batch_size):
            node_feats_torch, adj_mats_torch, _ = \
                get_tensors(node_feats, adj_mats, switch_mask)

            # feed forward policy net
            switch_log_pi, switch_pi = policy_net(
                node_feats_torch, adj_mats_torch)

            # sample action
            switch_p = Categorical(switch_pi)
            switch_a = switch_p.sample().item()

            # convert to masked version of the action
            if switch_a not in (traj_gen.env.switches_to_update - \
                                traj_gen.env.intermediate_switches):
                masked_a = traj_gen.env.num_switches
            else:
                masked_a = switch_a

            next_node_feats, next_adj_mats, next_switch_mask, \
                reward, done = traj_gen.step(masked_a)

            # store into storage
            for i in range(len(batch_node_feats)):
                batch_node_feats[i][ba, :, :] = node_feats[i][:, :]
                batch_next_node_feats[i][ba, :, :] = next_node_feats[i][:, :]
                batch_adj_mats[i][ba, :, :] = adj_mats[i][:, :]
                batch_next_adj_mats[i][ba, :, :] = next_adj_mats[i][:, :]
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
        batch_actions_torch = torch.from_numpy(batch_actions)
        batch_rewards_torch = torch.from_numpy(batch_rewards)
        batch_dones_torch = torch.from_numpy(batch_dones)
        batch_states_torch = (batch_node_feats_torch, batch_adj_mats_torch)

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
        adv = gae_advantage(batch_rewards, batch_dones, values_np,
            next_values_np, config.gamma, config.lam,
            norm=config.adv_norm)
        adv = torch.from_numpy(adv)
        
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
        monitor_reward = get_monitor_total_rewards(batch_rewards, batch_dones)
        monitor.add_scalar('Loss/pg_loss', pg_loss, epoch)
        monitor.add_scalar('Loss/v_loss', v_loss, epoch)
        monitor.add_scalar('Reward/avg_reward', np.mean(monitor_reward), epoch)
        monitor.add_scalar('Reward/min_reward', min(monitor_reward), epoch)
        monitor.add_scalar('Reward/max_reward', max(monitor_reward), epoch)
        monitor.add_histogram('Reward/sum_rewards', np.array(monitor_reward), epoch)
        monitor.add_scalar('Entropy/norm_entropy',
            entropy / - np.log(config.num_switches + 1), epoch)
        monitor.add_scalar('Entropy/entropy_factor', entropy_factor, epoch)
        monitor.add_scalar('Time/elapsed', proj_progress.delta_time, epoch)

        # save model, do testing
        if epoch % config.model_saving_interval == 0:
            torch.save(policy_net.state_dict(), config.result_folder +
                'policy_net_epoch_{}'.format(epoch))
            torch.save(value_net.state_dict(), config.result_folder +
                'value_net_epoch_{}'.format(epoch))


if __name__ == '__main__':
    main()
