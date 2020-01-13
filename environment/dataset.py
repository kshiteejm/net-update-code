import random
import numpy as np
import os, sys
import networkx as nx

from helpers import powerset, get_baseline_bw_matrix
from fat_tree_network import FatTreeNetwork
from traffic_distribution import TrafficDistribution
from waterfilling import MaxMinFairBW
from cost_function import CostFunction
sys.path.append(os.path.abspath('../'))
from proj_time import ProjectFinishTime

class Dataset:
    def __init__(self, pods=4, link_bw=10000.0):
        # init network and update switch set
        self.network = FatTreeNetwork(pods=pods, link_bw=link_bw)
        
        # init traffic matrix between tor pairs
        self.traffic_distribution = TrafficDistribution(self.network.num_tor_switches)
        # self.traffic_matrix = self.traffic_distribution.uniform(mean_min=1875, mean_max=1875, 
        #                                                    spread=625)
        # self.traffic_matrix = self.traffic_distribution.uniform(mean_min=2375, mean_max=2375, 
        #                                                    spread=625)
        self.traffic_matrix = self.traffic_distribution.uniform(mean_min=1875, mean_max=1875, 
                                                           spread=1200)
        print(np.sum(self.traffic_matrix))

        # waterfilling algorithm for max-min fair bw calculation
        self.max_min_fair_bw_calculator = MaxMinFairBW(self.network, self.traffic_matrix)

        # cost-function initialization
        baseline_bw_matrix = get_baseline_bw_matrix(self.max_min_fair_bw_calculator)
        self.cost_function = CostFunction('linear_relative', 
                                          baseline_bw_matrix, self.network.bisection_bw)

    # generate all possible one-step update costs
    def generate_costs(self, cost_file_name):
        cost_file = open(cost_file_name, 'w')
        cost_file.write("cost,down_idx\n")
        
        for switch_set in powerset(switch_set=self.network.update_switch_set):
            updated_bw_matrix = self.max_min_fair_bw_calculator. \
                                     get_traffic_class_fair_bw_matrix(switch_set)
            cost = self.cost_function.get_cost(updated_bw_matrix)
            
            switch_set_string = ""
            for switch in sorted(switch_set):
                switch_set_string =  switch_set_string + str(switch) + ","
            switch_set_string = switch_set_string[:-1]
            
            cost_file.write("%s,%s\n" % (round(cost, 2), switch_set_string))
          
        cost_file.close()

    # generate training dataset for training the gcn
    def generate_gcn_dataset(self, optimal_cost_action_file, 
                             save_nodefeats_file, save_adjmats_file, save_cost_file, 
                             max_num_steps):
        # generate a quad-graph - with 4 types of nodes
        # Traffic Class (TC) <-> Paths (P) <-> Links(L) <-> Switches (S)
        # TC raw feat = {traffic demand}
        # P raw feat = {}
        # L raw feat = {capacity, used-capacity}
        # S raw feat = {}   

        f = open(optimal_cost_action_file, 'r')
        gcn_dataset = []
        for line in f.readlines():
            line = line[:-1]
            triplet = [x.strip() for x in line.split(',')]
            optimal_remaining_cost = float(triplet[0])
            num_steps_left = float(triplet[1])
            update_switch_set_left = set()
            for i in range(2, len(triplet)):
                update_switch_set_left.add(int(triplet[i]))
            gcn_dataset.append((num_steps_left, optimal_remaining_cost, 
                                update_switch_set_left))
        f.close()

        proj_done_time = ProjectFinishTime(len(gcn_dataset), same_line=False)
        datum_id = 1
        for gcn_datum in gcn_dataset:
            num_steps = gcn_datum[0]
            total_cost = gcn_datum[1]
            update_switch_set = gcn_datum[2]

            # create nodes and register node-indices for each node type
            graph = self.network.graph
            q_graph = nx.Graph()
            tc = [] # traffic classes
            p = [] # paths
            l = [] # links
            s = [] # switches
            link_node_dict = dict()
            tc_node_dict = dict()
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
                q_graph.add_node(node_id, type='s', id=node, 
                                 raw_feats=[to_update,num_steps/max_num_steps])
                s.append(node_id)
                node_id = node_id + 1
            # initialize l
            for edge in graph.edges:
                q_graph.add_node(node_id, type='l', id=edge, 
                                 raw_feats=[
                                 graph.edges[edge]['capacity']/self.network.link_bw, 
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
                                                raw_feats=[0.0,0.0])
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
                                         0.0])
                        for p_node_id in tc_node_dict[(src, dst)]:
                            q_graph.add_edge(node_id, p_node_id)
                        tc.append(node_id)
                        node_id = node_id + 1
            # number of total nodes * 2 
            s_node_features = np.zeros((len(q_graph.nodes), 2))
            l_node_features = np.zeros((len(q_graph.nodes), 2))
            p_node_features = np.zeros((len(q_graph.nodes), 2))
            tc_node_features = np.zeros((len(q_graph.nodes), 2))
            for node_id in s:
                s_node_features[node_id][0] = q_graph.nodes[node_id]['raw_feats'][0]
                s_node_features[node_id][1] = q_graph.nodes[node_id]['raw_feats'][1]
            for node_id in l:
                l_node_features[node_id][0] = q_graph.nodes[node_id]['raw_feats'][0]
                l_node_features[node_id][1] = q_graph.nodes[node_id]['raw_feats'][1]
            for node_id in p:
                p_node_features[node_id][0] = q_graph.nodes[node_id]['raw_feats'][0]
                p_node_features[node_id][1] = q_graph.nodes[node_id]['raw_feats'][1]
            for node_id in tc:
                tc_node_features[node_id][0] = q_graph.nodes[node_id]['raw_feats'][0]
                tc_node_features[node_id][1] = q_graph.nodes[node_id]['raw_feats'][1]

            s_adj_matrix = nx.adjacency_matrix(q_graph).todense()
            l_adj_matrix = nx.adjacency_matrix(q_graph).todense()
            p_adj_matrix = nx.adjacency_matrix(q_graph).todense()
            tc_adj_matrix = nx.adjacency_matrix(q_graph).todense()        
            all_nodes = set(q_graph.nodes)
            for i in all_nodes.difference(set(s)):
                s_adj_matrix[i] = 0
            for i in all_nodes.difference(set(l)):
                l_adj_matrix[i] = 0
            for i in all_nodes.difference(set(p)):
                p_adj_matrix[i] = 0
            for i in all_nodes.difference(set(tc)):
                tc_adj_matrix[i] = 0

            np.save("%s_%s_%s" % (save_nodefeats_file, int(num_steps), datum_id), 
                [s_node_features, l_node_features, p_node_features, tc_node_features])
            np.save("%s_%s_%s" % (save_adjmats_file, int(num_steps), datum_id), 
                [s_adj_matrix, l_adj_matrix, p_adj_matrix, tc_adj_matrix])
            np.save("%s_%s_%s" % (save_cost_file, int(num_steps), datum_id),
                [total_cost])
            
            proj_done_time.update_progress(datum_id, message="elapsed")
            datum_id = datum_id + 1

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("python3 dataset.py \
                       seed \
                       gen_costs_bool \
                       gen_action_seq \
                       gen_visuals_bool \
                       dataset_loc")
    seed = sys.argv[1]
    generate_cost_file = ("True" == sys.argv[2])
    generate_action_seq = ("True" == sys.argv[3])
    generate_visualizations = ("True" == sys.argv[4])
    rust_dp = "../rust-dp"
    if len(sys.argv) > 5:
        dataset = sys.argv[5]
    else:
        dataset = "../data"
    pods = 4
    max_num_steps = 4
    total_cost = 0.0

    random.seed(seed)
    dataset_generator = Dataset(pods=pods)

    if generate_cost_file:
        dataset_generator.generate_costs("%s/costs_fat_tree_%s_pods_%s.csv" 
                                         % (dataset, pods, seed))

    if generate_action_seq:
        os.system("cd %s; \
                  ./target/debug/rust-dp --num-nodes 20 --num-steps %s \
                  --update-idx 0 1 2 3 4 5 8 9 12 13 16 17 \
                  --cm-path %s/costs_fat_tree_%s_pods_%s.csv \
                  --action-seq-path %s/action_seq_%s_pods_%s.csv \
                  --action-path %s/actions_%s_pods_%s.csv \
                  --value-path %s/values_%s_pods_%s.csv" 
                  % (rust_dp, max_num_steps, dataset, pods, seed, dataset, pods, seed,
                  dataset, pods, seed, dataset, pods, seed))
    
    optimal_cost_action_file = "%s/values_%s_pods_%s.csv" % (dataset, pods, seed)
    # if generate_visualizations:
    #     fat_tree_network.generate_visualization(
    #                         "%s/action_seq_%s_pods_%s.csv" 
    #                         % (dataset, pods, seed), 
    #                         "%s/graph_fat_tree_%s_pods_%s" 
    #                         % (dataset, pods, seed),
    #                         optimal_cost_action_file, 
    #                         max_num_steps)

    save_nodefeats_file =  "%s/nodefeats_fat_tree_%s_pods_%s" % (dataset, pods, seed)
    save_adjmats_file =  "%s/adjmats_fat_tree_%s_pods_%s" % (dataset, pods, seed)
    save_cost_file = "%s/cost_fat_tree_%s_pods_%s" % (dataset, pods, seed)
    dataset_generator.generate_gcn_dataset(optimal_cost_action_file, save_nodefeats_file, 
                                 save_adjmats_file, save_cost_file, max_num_steps)
