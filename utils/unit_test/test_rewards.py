import unittest
import numpy as np
from param import config
from utils.rewards import get_monitor_total_rewards


class TestRewards(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_monitor_rewards(self):
        avg_total_reward = get_monitor_total_rewards(
            [0,0,4,0,0,0,3,0,0,2,0,1],
            [0,0,0,0,0,0,1,0,0,0,0,1])
        assert(np.isclose(avg_total_reward, 5, config.eps))