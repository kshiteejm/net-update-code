import unittest
import numpy as np
from param import config
from environment.dcn_environment import DCNEnvironment


class TestDCNEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_reward(self):
        np.random.seed(config.seed)
        dcn_env = DCNEnvironment(pods=4, link_bw=10000.0, max_num_steps=4)
        down_switch_set = dcn_env.get_update_switch_set()
        updated_traffic_matrix = dcn_env\
                                 .max_min_fair_bw_calculator\
                                 .get_traffic_class_fair_bw_matrix(down_switch_set)
        cost = dcn_env.cost_function.get_cost(updated_traffic_matrix)
        assert(np.isclose(0.6379625, cost, config.eps))