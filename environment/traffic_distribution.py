import random
import numpy as np

class TrafficDistribution:
    def __init__(self, num_endpoints):
        self.num_endpoints = num_endpoints

    def uniform(self, mean_min=1250, mean_max=2500, spread=625):
        avg_tor_pair_traffic = random.randint(mean_min, mean_max)
        minima = avg_tor_pair_traffic - spread
        maxima = avg_tor_pair_traffic + spread
        num_endpoints = self.num_endpoints
        traffic_matrix = np.zeros((num_endpoints, num_endpoints))
        for src in range(num_endpoints):
            for dst in range(num_endpoints):
                if src == dst:
                    continue
                traffic_matrix[src][dst] = float(random.randint(minima, maxima))
        return traffic_matrix