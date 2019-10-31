import networkx as nx
from networkx import bipartite
import random
from itertools import combinations, chain
import numpy as np
import sys, traceback

class FatTreeNetwork:
    def __init__(self, pods=4, link_bw=10000):
        self.pods = pods
        
        graph = nx.DiGraph()

        num_tor_switches = (pods*pods)//2
        num_agg_switches = (pods*pods)//2
        num_core_switches = (pods//2) ** 2
        num_switches = num_tor_switches + \
                       num_agg_switches + \
                       num_core_switches
        
        graph.add_nodes_from(range(num_switches))

        link_id = 0
        for pod in range(pods):
            core_switch = 0
            for agg_offset in range(pods//2):
                agg_switch = num_core_switches + \
                             pod*pods + agg_offset
                # core - agg links; each agg is connected to pods//2 core switches
                for _ in range(pods//2):
                    graph.add_edge(core_switch, agg_switch, 
                        id=link_id, capacity=link_bw)
                    link_id = link_id + 1
                    graph.add_edge(agg_switch, core_switch, 
                        id=link_id, capacity=link_bw)
                    link_id = link_id + 1
                    core_switch += 1
                # agg - tor links; each pod has pods//2 agg and pods//2 tor switches
                for tor_offset in range(pods//2, pods):
                    tor_switch = num_core_switches + \
                                pod*pods + tor_offset
                    graph.add_edge(agg_switch, tor_switch, 
                        id=link_id, capacity=link_bw)
                    link_id = link_id + 1
                    graph.add_edge(tor_switch, agg_switch, 
                        id=link_id, capacity=link_bw)
                    link_id = link_id + 1

        self.graph = graph
        self.num_tor_switches = num_tor_switches
        self.num_agg_switches = num_agg_switches
        self.num_core_switches = num_core_switches
        self.num_switches = num_switches

        self.traffic_matrix = np.zeros((num_tor_switches, num_tor_switches))
        for src in range(num_tor_switches):
            for dst in range(num_tor_switches):
                if src == dst:
                    continue
                self.traffic_matrix[src][dst] = random.randint(2500, 7500)

        self.update_switch_set = set()
        for core_logical_id in range(self.num_core_switches):
            self.update_switch_set.add(self._get_core_physical_id(core_logical_id))
        for agg_logical_id in range(self.num_agg_switches):
            self.update_switch_set.add(self._get_agg_physical_id(agg_logical_id))
        
        # print(graph.nodes)
        # print(graph.edges)
        # print(self.traffic_matrix)

    def _get_tor_physical_id(self, tor_logical_id):
        pod = (tor_logical_id + 1) // self.pods
        tor_offset = (tor_logical_id % (self.pods//2)) + self.pods//2
        tor_physical_id = self.num_core_switches + pod*self.pods + tor_offset
        return tor_physical_id

    def _get_agg_physical_id(self, agg_logical_id):
        pod = (agg_logical_id + 1) // self.pods
        agg_offset = (agg_logical_id % (self.pods//2))
        agg_physical_id = self.num_core_switches + pod*self.pods + agg_offset
        return agg_physical_id

    def _get_core_physical_id(self, core_logical_id):
        return core_logical_id

    def powerset(self, switch_set):
        s = list(switch_set)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    def generate_bi_graph(self, switch_set):
        remaining_switch_set = set(self.graph.nodes).difference(set(switch_set))
        subgraph = self.graph.subgraph(remaining_switch_set)
        flows = []
        for src in range(self.num_tor_switches):
            for dst in range(self.num_tor_switches):
                if self.traffic_matrix[src][dst] > 0:
                    physical_src = self._get_tor_physical_id(src)
                    physical_dst = self._get_tor_physical_id(dst)
                    if nx.has_path(subgraph, physical_src, physical_dst):
                        for path in nx.shortest_simple_paths(subgraph, physical_src, physical_dst):
                            flows.append([src, dst, path])
                            # print(physical_src)
                            # print(physical_dst)
                            # print(path)

        bi_graph = nx.Graph()
        for flow in flows: 
            src = flow[0]
            dst = flow[1]
            path = flow[2]

            flow_node = (src, dst, 'flow')
            bi_graph.add_node(flow_node, bipartite=1)
            
            link_node = (src, src, dst)
            bi_graph.add_node(link_node, bipartite=0, capacity=self.traffic_matrix[src][dst])
            bi_graph.add_edge(link_node, flow_node)

            link_node = (dst, dst, src)
            bi_graph.add_node(link_node, bipartite=0, capacity=self.traffic_matrix[src][dst])
            bi_graph.add_edge(link_node, flow_node)

            for j in range(1, len(path)):
                i = j - 1
                link_node = (path[i], path[j], 'link')
                bi_graph.add_node(link_node, bipartite=0, capacity=subgraph[path[i]][path[j]]['capacity'])
                bi_graph.add_edge(link_node, flow_node)
        
        # print(bi_graph.nodes)
        # print(bi_graph.edges)

        return bi_graph
        
    def get_cost(self, bi_graph, baseline_bw):
        max_min_bw = self.get_max_min_bw(bi_graph)
        cost = 0
        for flow_node in baseline_bw:
            if flow_node not in max_min_bw:
                cost = cost + baseline_bw[flow_node]
            else: 
                cost = cost + abs(baseline_bw[flow_node] - max_min_bw[flow_node])
        
        return cost

    def get_max_min_bw(self, bi_graph):
        fair_bw = dict() 
        if bi_graph.size() == 0:
            return fair_bw

        while bi_graph.number_of_edges() > 0: 
            min_link_node = None
            min_bottleneck_bw = sys.maxsize

            link_nodes = [node for node in bi_graph.nodes if bi_graph.nodes[node]['bipartite'] == 0]
            flow_nodes = [node for node in bi_graph.nodes if bi_graph.nodes[node]['bipartite'] == 1]
                
            for link_node in link_nodes:
                # print(bi_graph[link_node])
                capacity = bi_graph.nodes[link_node]['capacity']
                flows = bi_graph.degree[link_node]
                bottleneck_bw = capacity/flows
                if bottleneck_bw < min_bottleneck_bw:
                    min_link_node = link_node
                    min_bottleneck_bw = bottleneck_bw
            min_link_flows = list(bi_graph.neighbors(min_link_node))
            bi_graph.remove_node(min_link_node)
            for flow_node in min_link_flows:
                fair_bw[flow_node] = min_bottleneck_bw
                flow_links = list(bi_graph.neighbors(flow_node))
                bi_graph.remove_node(flow_node)
                for link_node in flow_links:
                    if bi_graph.degree[link_node] == 0:
                        bi_graph.remove_node(link_node)
                        continue
                    bi_graph.nodes[link_node]['capacity'] = bi_graph.nodes[link_node]['capacity'] - min_bottleneck_bw

        
        return fair_bw

    def generate_costs(self):
        baseline_bi_graph = self.generate_bi_graph(set())
        baseline_bw = self.get_max_min_bw(baseline_bi_graph)
        print(baseline_bw)
        
        for switch_set in self.powerset(self.update_switch_set):
            bi_graph = self.generate_bi_graph(switch_set)
            cost = self.get_cost(bi_graph, baseline_bw)
            print(cost, switch_set)

if __name__ == '__main__':
    random.seed(10)
    fat_tree_network = FatTreeNetwork()
    fat_tree_network.generate_costs()
