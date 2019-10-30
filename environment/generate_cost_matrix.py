import numpy as np
import networkx as nx
import random
import sys

class RandomGraph():
    def __init__(self, num_nodes=4):#, cost_fn, traffic_matrix):
        self.graph = nx.gnp_random_graph(num_nodes, 0.5)

        # add link capacities
        for node in self.graph.nodes:
            self.graph.nodes[node]['capacity']  = random.randint(101, 1001)

        # for src, dst in traffic_matrix:
