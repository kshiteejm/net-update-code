import time
from environment.rl_interface import RLEnv
from gcn.batch_mgcn_policy import Batch_MGCN_Policy
from gcn.batch_mgcn_value import Batch_MGCN_Value


def main():
    policy_net = Batch_MGCN_Policy(20, [3,3,3,3], 8, [16,32], 8, 8)
    value_net = Batch_MGCN_Value(20, [3,3,3,3], 8, [16,32], 8, 8)
    env = RLEnv()
    state = env.reset()
    # convert state
    switch_logpi, switch_pi, masked_pi = policy_net(
        node_feats_torch, adj_mats_torch, switch_mask)

    switch_p = Categorical(masked_pi)
    switch_a = switch_p.sample().item()

    env.step(switch_a)


if __name__ == '__main__':
    main()
