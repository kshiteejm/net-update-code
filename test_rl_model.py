import config

def main(model_dir, n_epoch):
    # policy network for taking actions and policy gradient
    policy_net = Batch_MGCN_Policy(
        config.num_switches, [config.class_feat, config.path_feat,
        config.link_feat, config.switch_feat], config.n_output,
        config.hid_dim, config.h_size, config.n_steps)
    state_dicts = torch.load("%s/policy_net_epoch_%s" % (model_dir, n_epoch))
    policy_net.load_state_dict(state_dicts)

    # value network for value prediction and advantage estimation
    value_net = Batch_MGCN_Value(
        config.num_switches, [config.class_feat, config.path_feat,
        config.link_feat, config.switch_feat], config.n_output,
        config.hid_dim, config.h_size, config.n_steps)
    state_dicts = torch.load("%s/value_net_epoch_%s" % (model_dir, n_epoch))
    value_net.load_state_dict(state_dicts)

if __name__ == '__main__':
    model_dir = sys.argv[1]
    n_epoch = sys.argv[2]
    main(model_dir, n_epoch)