import argparse

parser = argparse.ArgumentParser(description='parameters')

# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')

# -- Environment --
parser.add_argument('--num_switches', type=int, default=20,
                    help='number of switches (default: 20)')
parser.add_argument('--class_feat', type=int, default=3,
                    help='number of features for traffic class (default: 3)')
parser.add_argument('--path_feat', type=int, default=3,
                    help='number of features for flow path (default: 3)')
parser.add_argument('--link_feat', type=int, default=3,
                    help='number of features for links (default: 3)')
parser.add_argument('--switch_feat', type=int, default=3,
                    help='number of features for switches (default: 3)')
parser.add_argument('--n_output', type=int, default=8,
                    help='number of intermediate features (default: 8)')
parser.add_argument('--hid_dim', type=int, default=[16,32], nargs='+',
                    help='number of hidden features (default: [16,32])')
parser.add_argument('--h_size', type=int, default=8,
                    help='message passing hidden features (default: 8)')
parser.add_argument('--n_steps', type=int, default=8,
                    help='number of message passing steps (default: 8)')

# -- Learning --
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for policy gradient (default: 64)')
parser.add_argument('--lr_rate', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--num_epochs', type=int, default=10000,
                    help='number of trianing iterations (default: 10000)')
parser.add_argument('--gamma', type=float, default=1,
                    help='discount factor (default: 1)')
parser.add_argument('--lam', type=float, default=1,
                    help='td lambda (default: 1)')

config, _ = parser.parse_known_args()