import networkx as nx
from networkx import bipartite
import random

from itertools import combinations, chain
import numpy as np
import sys, traceback

from graphviz import Graph
from IPython.display import display

class FatTreeNetwork:
    def __init__(self, pods=4, link_bw=10000, generate_cost_file=True):
        self.pods = pods
        self.link_bw = link_bw
        self.generate_cost_file = generate_cost_file

        graph = nx.DiGraph()

        num_tor_switches = (pods*pods)//2
        num_agg_switches = (pods*pods)//2
        num_core_switches = (pods//2) ** 2
        num_switches = num_tor_switches + \
                       num_agg_switches + \
                       num_core_switches
        
        graph.add_nodes_from(range(num_switches), active=True)

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
        
        print(self.traffic_matrix)

    def _get_tor_physical_id(self, tor_logical_id):
        pod = tor_logical_id // (self.pods//2)
        tor_offset = (tor_logical_id % (self.pods//2)) + self.pods//2
        tor_physical_id = self.num_core_switches + pod*self.pods + tor_offset
        return tor_physical_id

    def _get_agg_physical_id(self, agg_logical_id):
        pod = agg_logical_id // (self.pods//2)
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
                        # shortest_path_len = sys.maxsize
                        for path in nx.all_shortest_paths(subgraph, physical_src, physical_dst):
                            flows.append([src, dst, path])
                            # print(src, dst, path)

        bi_graph = nx.Graph()
        for flow in flows: 
            src = flow[0]
            dst = flow[1]
            path = flow[2]

            flow_node = (src, dst, 'flow')
            bi_graph.add_node(flow_node, bipartite=1)
            
            link_node = (src, dst, src)
            bi_graph.add_node(link_node, bipartite=0, capacity=self.traffic_matrix[src][dst])
            bi_graph.add_edge(link_node, flow_node)

            link_node = (src, dst, dst)
            bi_graph.add_node(link_node, bipartite=0, capacity=self.traffic_matrix[src][dst])
            bi_graph.add_edge(link_node, flow_node)

            for j in range(1, len(path)):
                i = j - 1
                link_node = (path[i], path[j], 'link')
                bi_graph.add_node(link_node, bipartite=0, capacity=subgraph[path[i]][path[j]]['capacity'])
                bi_graph.add_edge(link_node, flow_node)
        
        # print(bi_graph.nodes)
        # print(bi_graph.edges)

        return bi_graph, flows
        
    def get_cost(self, max_min_bw_matrix, baseline_bw_matrix):
        cost = np.sum(abs(baseline_bw_matrix - max_min_bw_matrix))
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
        if self.generate_cost_file:
            cost_file_name = "cost_fat_tree_%s_pods.csv" % (self.pods) 
            cost_file = open(cost_file_name, 'w')
        
        baseline_bi_graph, baseline_flows = self.generate_bi_graph(set())
        baseline_bw = self.get_max_min_bw(baseline_bi_graph)
        baseline_bw_matrix = self.get_flow_bw_matrix(baseline_bw)
        graph = self.get_updated_graph(baseline_flows, baseline_bw, set())

        print(baseline_bw_matrix)
        print(np.sum(baseline_bw_matrix))

        for switch_set in self.powerset(self.update_switch_set):
            bi_graph, flows = self.generate_bi_graph(switch_set)
            max_min_bw = self.get_max_min_bw(bi_graph)
            max_min_bw_matrix = self.get_flow_bw_matrix(max_min_bw)
            cost = self.get_cost(max_min_bw_matrix, baseline_bw_matrix)
            
            switch_set_string = ""
            for switch in switch_set:
                switch_set_string = str(switch) + "," + switch_set_string
            switch_set_string = switch_set_string[:-1]
            
            saved_file_name = "graph_%s_%s.yaml" % (switch_set_string, round(cost, 2))

            if self.generate_cost_file:
                cost_file.write("%s, %s\n" % (round(cost, 2), switch_set_string))
            print("%s, %s, %s" % (round(cost, 2), switch_set_string, saved_file_name))
            
            # graph = self.get_updated_graph(flows, max_min_bw, switch_set)
            # nx.write_yaml(graph, saved_file_name)
            
            if (switch_set_string == "16,5,1"):
                graph = self.get_updated_graph(flows, max_min_bw, switch_set)
                self.visualize_graph(graph, cost)
                
        if self.generate_cost_file:
            cost_file.close()

    def get_flow_bw_matrix(self, flow_bw):
        flow_bw_matrix = np.zeros((self.num_tor_switches, self.num_tor_switches))
        for flow in flow_bw:
            src = flow[0]
            dst = flow[1]
            flow_bw_matrix[src][dst] = flow_bw_matrix[src][dst] + flow_bw[flow]
        return flow_bw_matrix

    def get_updated_graph(self, flows, flow_bw, switch_set):
        graph = self.graph.copy()

        for switch in switch_set:
            graph.nodes[switch]['active'] = False

        for flow in flows:
            src = flow[0]
            dst = flow[1]
            path = flow[2]
            flow_node = (src, dst, 'flow')
            bw = flow_bw[flow_node]
            for j in range(1, len(path)):
                i = j - 1
                path_src = path[i]
                path_dst = path[j]
                graph[path_src][path_dst]['used_capacity'] = \
                    graph[path_src][path_dst]['used_capacity'] + bw

        return graph
    
    def visualize_graph(self, graph, cost):
        core_physical_ids = [self._get_core_physical_id(logical_id) 
                            for logical_id in range(0, self.num_core_switches)]
        agg_physical_ids = [self._get_agg_physical_id(logical_id) 
                            for logical_id in range(0, self.num_agg_switches)]
        tor_physical_ids = [self._get_tor_physical_id(logical_id) 
                            for logical_id in range(0, self.num_tor_switches)]
        
        physical_ids = [core_physical_ids, agg_physical_ids, tor_physical_ids]

        # TODO: specify better max cap
        pen_scale = 5000

        # TODO: automate vertical coordinate
        v_dict = {0: 2, 1: 1, 2: 0}

        # generate coordinate for nodes
        h_max = max(len(l) for l in physical_ids)
        coord_dict = {}
        for i in range(len(physical_ids)):
            h = len(physical_ids[i])
            offset = int((h_max - h) / 2)
            for j in range(len(physical_ids[i])):
                physical_id = physical_ids[i][j]
                coord_dict[physical_id] = \
                (j + offset , v_dict[i])  # looks better
        
        # graph visualization object (pos needs neato)
        vg = Graph('switch_graph', engine='neato')

        # place nodes in the graph
        for n in graph.nodes:
            if graph.nodes[n]['active']:
                color = 'white'
            else:
                color = 'dimgray'
            vg.node(str(n), pos='{}, {}!'.format(
                coord_dict[n][0], coord_dict[n][1]),
                shape='circle', fillcolor=color,
                style='filled')

        # TODO: (check) prevent replotting the same edge
        edge_record = set()

        # visualize edges
        for e in graph.edges:
            src = e[0]
            dst = e[1]
            if (dst, src) not in edge_record:
                edge_record.add((src, dst))
            else:
                # check 'used_capacity' is the same
                np.isclose(graph.edges[e]['used_capacity'],
                        graph.edges[(dst, src)]['used_capacity'])
                continue
            vg.edge(str(src), str(dst), penwidth=str(
                graph.edges[e]['used_capacity'] / pen_scale))

        # put a text for cost in an invisible node
        vg.node('Cost: {}'.format(cost),
                pos='0.5, {}!'.format(max(v_dict.keys()) + 0.5),
                color='white')

        vg.render('./data/visualization', view=True)

if __name__ == '__main__':
    random.seed(42)
    fat_tree_network = FatTreeNetwork(pods=4, generate_cost_file=False)
    fat_tree_network.generate_costs()
