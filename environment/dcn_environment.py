import random
import numpy as np
import os, sys
import networkx as nx

from environment.helpers import powerset, get_baseline_bw_matrix
from environment.fat_tree_network import FatTreeNetwork
from environment.traffic_distribution import TrafficDistribution
from environment.waterfilling import MaxMinFairBW
from environment.cost_function import CostFunction
import scipy.sparse
sys.path.append(os.path.abspath('../'))
from utils.proj_time import ProjectFinishTime

class DCNEnvironment:
    def __init__(self, pods=4, link_bw=10000.0, max_num_steps=4, 
                 mean_min=1875, mean_max=1875, spread=625):
        self.max_num_steps = 4

        # init network and update switch set
        self.network = FatTreeNetwork(pods=pods, link_bw=link_bw)
        
        # init traffic matrix between tor pairs
        self.traffic_distribution = TrafficDistribution(self.network.num_tor_switches)
        self.traffic_matrix = self.traffic_distribution.uniform(mean_min=mean_min, 
                                                                mean_max=mean_max, 
                                                                spread=spread)

        # waterfilling algorithm for max-min fair bw calculation
        self.max_min_fair_bw_calculator = MaxMinFairBW(self.network, self.traffic_matrix)

        # cost-function initialization
        baseline_bw_matrix = get_baseline_bw_matrix(self.max_min_fair_bw_calculator)
        self.cost_function = CostFunction('linear_relative', 
                                          baseline_bw_matrix, self.network.bisection_bw)

    def get_update_switch_set(self):
        return self.network.update_switch_set
    
    def get_max_num_steps(self):
        return self.max_num_steps

    def get_total_switches(self):
        return self.network.num_switches
    
    # generate all possible one-step update costs
    def get_cost_model(self):
        cost_model = {}
        for switch_set in powerset(switch_set=self.network.update_switch_set):
            updated_bw_matrix = self.max_min_fair_bw_calculator. \
                                     get_traffic_class_fair_bw_matrix(switch_set)
            cost = self.cost_function.get_cost(updated_bw_matrix)
            cost_model[tuple(sorted(switch_set))] = cost
        return cost_model
    
    # generate all possible one-step update costs
    def generate_costs(self, cost_file_name):
        cost_file = open(cost_file_name, 'w')
        cost_file.write("cost,down_idx\n")
        
        # cost_file_rows = (2 ** len(self.network.update_switch_set))
        # print("powerset size: %s" % cost_file_rows)
        # proj_done_time = ProjectFinishTime(cost_file_rows, same_line=False)
        # row = 0
        for switch_set in powerset(switch_set=self.network.update_switch_set):
            updated_bw_matrix = self.max_min_fair_bw_calculator. \
                                     get_traffic_class_fair_bw_matrix(switch_set)
            cost = self.cost_function.get_cost(updated_bw_matrix)
            
            switch_set_string = ""
            for switch in sorted(switch_set):
                switch_set_string =  switch_set_string + str(switch) + ","
            switch_set_string = switch_set_string[:-1]
            
            cost_file.write("%s,%s\n" % (round(cost, 2), switch_set_string))
            
            # row = row + 1
            # proj_done_time.update_progress(row, message="elapsed")
          
        cost_file.close()

    def get_state(self, update_switch_set, intermediate_switches, num_steps):
        # create nodes and register node-indices for each node type
        graph = self.network.graph
        q_graph = nx.Graph()
        tc = [] # traffic classes
        p = [] # paths
        l = [] # links
        s = [] # switches
        link_node_dict = dict()
        tc_node_dict = dict()
        raw_feats_size = 3
        node_id = 0

        # initialize tc_node_dict
        for src in range(self.network.num_tor_switches):
            for dst in range(self.network.num_tor_switches):
                if src != dst:
                    tc_node_dict[(src, dst)] = []
        
        # initialize s
        for node in graph.nodes:
            to_update = 0.0
            if node in update_switch_set:
                to_update = 1.0
            intermediate = 0.0
            if node in intermediate_switches:
                intermediate = 1.0
            q_graph.add_node(node_id, type='s', id=node, 
                                raw_feats=[
                                intermediate,
                                to_update,
                                num_steps/self.max_num_steps])
            s.append(node_id)
            node_id = node_id + 1
        
        # initialize l
        for edge in graph.edges:
            q_graph.add_node(node_id, type='l', id=edge, 
                                raw_feats=[
                                graph.edges[edge]['capacity']/self.network.link_bw, 
                                0.0,
                                0.0])
            q_graph.add_edge(node_id, edge[0])
            q_graph.add_edge(node_id, edge[1])
            l.append(node_id)
            link_node_dict[edge] = node_id
            node_id = node_id + 1
        
        # initialize p
        for src in range(self.network.num_tor_switches):
            for dst in range(self.network.num_tor_switches):
                if src != dst:
                    physical_src = self.network.get_tor_physical_id(src)
                    physical_dst = self.network.get_tor_physical_id(dst)
                    if nx.has_path(graph, physical_src, physical_dst):
                        path_num = 0
                        for path in nx.all_shortest_paths(graph, 
                                        physical_src, physical_dst):
                            q_graph.add_node(node_id, type='p', id=(src,dst,path_num), 
                                            raw_feats=[0.0,0.0,0.0])
                            for j in range(1, len(path)):
                                i = j - 1
                                path_src = path[i]
                                path_dst = path[j]
                                q_graph.add_edge(node_id, 
                                        link_node_dict[(path_src, path_dst)])
                            p.append(node_id)
                            tc_node_dict[(src, dst)].append(node_id)
                            node_id = node_id + 1
                            path_num = path_num + 1
        
        # initialize tc
        for src in range(self.network.num_tor_switches):
            for dst in range(self.network.num_tor_switches):
                if src != dst:
                    q_graph.add_node(node_id, type='tc', id=(src,dst), 
                                        raw_feats=[
                                        self.traffic_matrix[src][dst]/self.network.link_bw, 
                                        0.0,
                                        0.0])
                    for p_node_id in tc_node_dict[(src, dst)]:
                        q_graph.add_edge(node_id, p_node_id)
                    tc.append(node_id)
                    node_id = node_id + 1
        
        # number of total nodes * 2 
        s_node_features = np.zeros((len(q_graph.nodes), raw_feats_size))
        l_node_features = np.zeros((len(q_graph.nodes), raw_feats_size))
        p_node_features = np.zeros((len(q_graph.nodes), raw_feats_size))
        tc_node_features = np.zeros((len(q_graph.nodes), raw_feats_size))
        for node_id in s:
            for feat_id in range(raw_feats_size):
                s_node_features[node_id][feat_id] = \
                    q_graph.nodes[node_id]['raw_feats'][feat_id]
        for node_id in l:
            for feat_id in range(raw_feats_size):
                l_node_features[node_id][feat_id] = \
                    q_graph.nodes[node_id]['raw_feats'][feat_id]
        for node_id in p:
            for feat_id in range(raw_feats_size):
                p_node_features[node_id][feat_id] = \
                    q_graph.nodes[node_id]['raw_feats'][feat_id]
        for node_id in tc:
            for feat_id in range(raw_feats_size):
                tc_node_features[node_id][feat_id] = \
                    q_graph.nodes[node_id]['raw_feats'][feat_id]

        adj_matrices = {}
        adj_matrices['s'] = nx.adjacency_matrix(q_graph).toarray()
        adj_matrices['l'] = nx.adjacency_matrix(q_graph).toarray()
        adj_matrices['p'] = nx.adjacency_matrix(q_graph).toarray()
        adj_matrices['tc'] = nx.adjacency_matrix(q_graph).toarray()
        all_nodes = set(q_graph.nodes)
        for i in all_nodes.difference(set(s)):
            adj_matrices['s'][i] = 0
        for i in all_nodes.difference(set(l)):
            adj_matrices['l'][i] = 0
        for i in all_nodes.difference(set(p)):
            adj_matrices['p'][i] = 0
        for i in all_nodes.difference(set(tc)):
            adj_matrices['tc'][i] = 0

        node_feats = [s_node_features, l_node_features, p_node_features, tc_node_features]
        adj_mats = [adj_matrices['s'], adj_matrices['l'], 
                    adj_matrices['p'], adj_matrices['tc']]

        switch_mask = np.zeros(self.network.num_switches + 1)
        switch_mask[-1] = 1
        for switch_id in (update_switch_set - intermediate_switches):
            switch_mask[switch_id] = 1.0

        return node_feats, adj_mats, switch_mask   
