import numpy as np
from itertools import combinations, chain


# generate a powerset except the empty set
def powerset(switch_set):
    s = list(switch_set)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

# generate baselines with all switches active
def get_baseline_bw_matrix(max_min_fair_bw_calculator):
    # generate baseline max-min fair bw allocation with all switches active
    baseline_bw_matrix = max_min_fair_bw_calculator.get_traffic_class_fair_bw_matrix(set())

    # print(baseline_bw_matrix)
    # print(np.sum(baseline_bw_matrix))
    
    return baseline_bw_matrix