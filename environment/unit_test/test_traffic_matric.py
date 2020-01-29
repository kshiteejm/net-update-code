import unittest
import numpy as np
from environment.traffic_distribution import TrafficDistribution

class TestTrafficMatrix(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def test_uniform_traffic_matrix(self):
        td = TrafficDistribution(20)
        td.uniform()
        pass

    def test_gravity_model_traffic_matrix(self):
        td = TrafficDistribution(20)
        T = 1000
        traffic_matrix = td.gravity_model(T)
        print(np.sum(traffic_matrix))
        assert(np.sum(traffic_matrix) < T)



