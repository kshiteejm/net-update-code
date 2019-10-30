import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import random
import sys

class BipartiteGraph():
    def __init__(self, num_u_nodes, num_v_nodes):
        self.bi_graph = bipartite.random_graph(num_u_nodes, num_v_nodes, 0.4)
        self.u_nodes = {n for n, d in self.bi_graph.nodes(data=True) if d['bipartite']==0}
        self.v_nodes = set(self.bi_graph) - self.u_nodes
        self.u_nodes = list(self.u_nodes)
        self.v_nodes = list(self.v_nodes)
        # ensure every v node has atleast one edge - translates to each flow atleast traversing one link
        for i in range(len(self.v_nodes)):
            self.bi_graph.add_edge(self.u_nodes[i%len(self.u_nodes)], self.v_nodes[i])

        for node in self.v_nodes:
            self.bi_graph.nodes[node]['node_feats'] = [0, 0]
        for node in self.u_nodes:
            self.bi_graph.nodes[node]['node_feats']  = [
                random.randint(101, 1001), self.bi_graph.degree(node), 1, 0, 0]
    
    def u_feats(self):
        return len(self.bi_graph.nodes[self.u_nodes[0]]['node_feats'])
    
    def v_feats(self):
        return len(self.bi_graph.nodes[self.v_nodes[0]]['node_feats'])
    
    def u_node_feats(self):
        u_node_feats = []
        for node in self.u_nodes:
            u_node_feats.append(self.bi_graph.nodes[node]['node_feats'])
        return np.array(u_node_feats)

    def v_node_feats(self):
        v_node_feats = []
        for node in self.v_nodes:
            v_node_feats.append(self.bi_graph.nodes[node]['node_feats'])
        return np.array(v_node_feats)

    def bi_adj_mat(self):
        uv_adj_mat = np.zeros((len(self.v_nodes), len(self.u_nodes)))
        vu_adj_mat = np.zeros((len(self.u_nodes), len(self.v_nodes)))
        for e in self.bi_graph.edges():
            u_node_index = self.u_nodes.index(e[0])
            v_node_index = self.v_nodes.index(e[1])
            uv_adj_mat[v_node_index][u_node_index] = 1.0
            vu_adj_mat[u_node_index][v_node_index] = 1.0
        return uv_adj_mat, vu_adj_mat
    
    def true_output(self):
        u_nodes = list(self.u_nodes)
        v_nodes = list(self.v_nodes)
        fair_bw = np.zeros(len(v_nodes))
        v_saturated_flows = set()
        u_completed_links = set()
        u_capacities = dict()
        u_flows = dict()

        u_vacuous_links = []
        for node in u_nodes:
            capacity = self.bi_graph.nodes[node]['node_feats'][0]
            num_flows = self.bi_graph.nodes[node]['node_feats'][1]
            if num_flows == 0:
                u_vacuous_links.append(node)
                continue
            u_capacities[node] = capacity
            u_flows[node] = num_flows
        for node in u_vacuous_links:
            u_nodes.remove(node)
            u_completed_links.add(node)

        while len(u_nodes) != 0 and len(v_nodes) != 0:
            min_u_node = None
            min_bottleneck_bw = sys.maxsize
            for node in u_nodes:
                bottleneck_bw = u_capacities[node]/u_flows[node]
                if bottleneck_bw < min_bottleneck_bw:
                    min_u_node = node
                    min_bottleneck_bw = bottleneck_bw
            
            min_u_node_ngbrs = set([n for n in self.bi_graph.neighbors(min_u_node)]) - v_saturated_flows
            for node in min_u_node_ngbrs:
                ngbrs = set([n for n in self.bi_graph.neighbors(node)]) - u_completed_links
                for node_u in ngbrs:
                    u_capacities[node_u] = u_capacities[node_u] - min_bottleneck_bw
                    u_flows[node_u] = u_flows[node_u] - 1
                    if u_flows[node_u] == 0:
                        u_nodes.remove(node_u)
                        u_completed_links.add(node_u)
                fair_bw[self.v_nodes.index(node)] = min_bottleneck_bw
                v_nodes.remove(node)
                v_saturated_flows.add(node)
            if min_u_node in u_nodes:
                u_nodes.remove(min_u_node)
                u_completed_links.add(min_u_node)
        
        return fair_bw

    def max_min_flow(self, bi_graph):
        u_nodes_original = {n for n, d in bi_graph.nodes(data=True) if d['bipartite']==0}
        v_nodes_original = set(self.bi_graph) - u_nodes_original
        u_nodes_original = list(u_nodes_original)
        v_nodes_original = list(v_nodes_original)

        for node in v_nodes_original:
            bi_graph.nodes[node]['node_feats'] = [0, 0]
        for node in u_nodes_original:
            bi_graph.nodes[node]['node_feats']  = [
                random.randint(101, 1001), self.bi_graph.degree(node), 1, 0, 0]
        
        u_nodes = list(u_nodes_original)
        v_nodes = list(v_nodes_original)
        fair_bw = np.zeros(len(v_nodes))
        v_saturated_flows = set()
        u_completed_links = set()
        u_capacities = dict()
        u_flows = dict()

        u_vacuous_links = []
        for node in u_nodes:
            capacity = bi_graph.nodes[node]['node_feats'][0]
            num_flows = bi_graph.nodes[node]['node_feats'][1]
            if num_flows == 0:
                u_vacuous_links.append(node)
                continue
            u_capacities[node] = capacity
            u_flows[node] = num_flows
        for node in u_vacuous_links:
            u_nodes.remove(node)
            u_completed_links.add(node)

        while len(u_nodes) != 0 and len(v_nodes) != 0:
            min_u_node = None
            min_bottleneck_bw = sys.maxsize
            for node in u_nodes:
                bottleneck_bw = u_capacities[node]/u_flows[node]
                if bottleneck_bw < min_bottleneck_bw:
                    min_u_node = node
                    min_bottleneck_bw = bottleneck_bw
            
            min_u_node_ngbrs = set([n for n in bi_graph.neighbors(min_u_node)]) - v_saturated_flows
            for node in min_u_node_ngbrs:
                ngbrs = set([n for n in bi_graph.neighbors(node)]) - u_completed_links
                for node_u in ngbrs:
                    u_capacities[node_u] = u_capacities[node_u] - min_bottleneck_bw
                    u_flows[node_u] = u_flows[node_u] - 1
                    if u_flows[node_u] == 0:
                        u_nodes.remove(node_u)
                        u_completed_links.add(node_u)
                fair_bw[v_nodes_original.index(node)] = min_bottleneck_bw
                v_nodes.remove(node)
                v_saturated_flows.add(node)
            if min_u_node in u_nodes:
                u_nodes.remove(min_u_node)
                u_completed_links.add(min_u_node)
        
        return fair_bw
        
        def estimate_linear_cost(self, update_switch_set):
            max_flows = self.true_output()
            self.bi_graph_updated = self.bi_graph.copy()
            for v_i in update_switch_set:
                self.bi_graph_updated.remove_node(v_i)
            updated_flows = self.max_min_flow()
            # for i in range(updated_flows):

            
            

