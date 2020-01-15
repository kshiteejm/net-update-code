import unittest
from environment.rl_interface import RLEnv


class TestEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_env_init(self):
        env = RLEnv()