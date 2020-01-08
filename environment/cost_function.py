import numpy as np
import networkx as nx
import random
import sys

class CostFunction:
    def __init__(self, type='linear_relative', *args):
        if type == 'linear_relative':
            self.baseline_cost = args[0]
            self.max_cost = args[1]
            self.cost_function = self.get_relative_cost

    def get_cost(self, *args):
        cost = self.cost_function(args)
        return cost

    # get the cost of a max min bw allocation on a graph with a subset of switches taken down
    # the cost function is a linear cost function at the moment
    def get_relative_cost(self, *args):
        baseline_bw_matrix = self.baseline_cost
        bisection_bw = self.max_cost
        updated_bw_matrix = args[0]

        cost = np.sum(abs(baseline_bw_matrix - updated_bw_matrix))
        cost = cost/bisection_bw

        return cost