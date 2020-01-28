import random
import numpy as np

class TrafficDistribution:
    def __init__(self, num_endpoints, total_traffic, endpoint_max):
        self.num_endpoints = num_endpoints

    def uniform(self, mean_min=1250, mean_max=2500, spread=625):
        avg_tor_pair_traffic = np.random.randint(mean_min, mean_max+1)
        minima = avg_tor_pair_traffic - spread
        maxima = avg_tor_pair_traffic + spread
        num_endpoints = self.num_endpoints
        traffic_matrix = np.zeros((num_endpoints, num_endpoints))
        for src in range(num_endpoints):
            for dst in range(num_endpoints):
                if src == dst:
                    continue
                traffic_matrix[src][dst] = float(np.random.randint(minima, maxima+1))
        return traffic_matrix

    def gravity_model(self, total_traffic):
        num_endpoints = self.num_endpoints
        T = total_traffic
        T_in = np.random.exponential(2, [num_endpoints])
        T_out = np.random.exponential(2, [num_endpoints])
        p_in = T_in/sum(T_in)
        p_out = T_out/sum(T_out)
        traffic_matrix = np.zeros((num_endpoints, num_endpoints))
        for src in range(num_endpoints):
            for dst in range(num_endpoints):
                if src == dst:
                    continue
                traffic_matrix[src][dst] = T * p_in[src] * p_out[dst]
        return traffic_matrix
