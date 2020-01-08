import networkx as nx
from networkx import bipartite
import random
import os

import numpy as np
import sys, traceback

from graphviz import Graph, Digraph
from IPython.display import display

from .traffic_distribution import TrafficDistribution

class FatTreeNetwork:
    def __init__(self, pods=4, link_bw=10000.0):
        self.pods = pods
        self.link_bw = link_bw

        graph = nx.DiGraph()

        # init switch cardinalities
        # (1 pod) contains (pods//2 agg switches, pods//2 tor switch)
        # (1 agg switch) connected to (pods//2 core switches)  
        num_tor_switches = (pods//2) * pods
        num_agg_switches = (pods//2) * pods
        num_core_switches = (pods//2) * (pods//2)
        num_switches = num_tor_switches + \
                       num_agg_switches + \
                       num_core_switches
        
        # init graph nodes
        graph.add_nodes_from(range(num_switches), active=True, step=-1)

        # init graph edges
        link_id = 0
        for pod in range(pods):
            core_switch = 0
            for agg_offset in range(pods//2):
                agg_switch = num_core_switches + \
                             pod*pods + agg_offset
                # core - agg links; each agg is connected to pods//2 core switches
                for _ in range(pods//2):
                    graph.add_edge(core_switch, agg_switch, 
                        id=link_id, capacity=link_bw, used_capacity=0)
                    link_id = link_id + 1
                    graph.add_edge(agg_switch, core_switch, 
                        id=link_id, capacity=link_bw, used_capacity=0)
                    link_id = link_id + 1
                    core_switch += 1
                # agg - tor links; each pod has pods//2 agg and pods//2 tor switches
                for tor_offset in range(pods//2, pods):
                    tor_switch = num_core_switches + \
                                pod*pods + tor_offset
                    graph.add_edge(agg_switch, tor_switch, 
                        id=link_id, capacity=link_bw, used_capacity=0)
                    link_id = link_id + 1
                    graph.add_edge(tor_switch, agg_switch, 
                        id=link_id, capacity=link_bw, used_capacity=0)
                    link_id = link_id + 1

        self.graph = graph
        self.num_tor_switches = num_tor_switches
        self.num_agg_switches = num_agg_switches
        self.num_core_switches = num_core_switches
        self.num_switches = num_switches

        # init set of switches to be updated in the network
        self.update_switch_set = set()
        for core_logical_id in range(self.num_core_switches):
            self.update_switch_set.add(self.get_core_physical_id(core_logical_id))
        for agg_logical_id in range(self.num_agg_switches):
            self.update_switch_set.add(self.get_agg_physical_id(agg_logical_id))

        self.bisection_bw = self.num_tor_switches * pods//2 * link_bw

    # helper functions to get a physical id from a logical id
    # physical id = node index in the networkx graph datastructure
    # logical id = [0 ... num_switches_of_that_type)
    def get_tor_physical_id(self, tor_logical_id):
        pod = tor_logical_id // (self.pods//2)
        tor_offset = (tor_logical_id % (self.pods//2)) + self.pods//2
        tor_physical_id = self.num_core_switches + pod*self.pods + tor_offset
        return tor_physical_id

    def get_agg_physical_id(self, agg_logical_id):
        pod = agg_logical_id // (self.pods//2)
        agg_offset = (agg_logical_id % (self.pods//2))
        agg_physical_id = self.num_core_switches + pod*self.pods + agg_offset
        return agg_physical_id

    def get_core_physical_id(self, core_logical_id):
        return core_logical_id

    def get_tor_physical_ids(self):
        tor_physical_ids = [self.get_tor_physical_id(logical_id) 
                            for logical_id in range(0, self.num_tor_switches)]
        return tor_physical_ids

    def get_agg_physical_ids(self):
        agg_physical_ids = [self.get_agg_physical_id(logical_id) 
                            for logical_id in range(0, self.num_agg_switches)]
        return agg_physical_ids

    def get_core_physical_ids(self):
        core_physical_ids = [self.get_core_physical_id(logical_id) 
                            for logical_id in range(0, self.num_core_switches)]
        return core_physical_ids
