import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import random
import sys

class JupiterGraph():
    def __init__(self, core, pod, agg_per_pod, tor_per_pod, link_bw):
        self.core = core
        self.pod = pod
        self.agg_per_pod = agg_per_pod
        self.tor_per_pod = tor_per_pod
        self.link_bw = link_bw
        self.graph = nx.Graph()

    def _num_switches(self):
        return (self.core + self.pod * (self.agg_per_pod + self.tor_per_pod))