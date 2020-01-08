import networkx as nx
import numpy as np
import sys

class MaxMinFairBW():
    def __init__(self, network, traffic_matrix):
        self.network = network
        self.traffic_matrix = traffic_matrix

    # generate traffic class i.e. src-dst tor pair max-min fair bw assignments
    def get_traffic_class_fair_bw_matrix(self, switch_set):
        bi_graph = self.generate_bi_graph(switch_set)
        path_bws = self.get_max_min_fair_bw(bi_graph)
        num_tor_switches = self.network.num_tor_switches

        traffic_class_bw_matrix = np.zeros((num_tor_switches, num_tor_switches))

        for path in path_bws:
            src = path[0]
            dst = path[1]
            traffic_class_bw_matrix[src][dst] = traffic_class_bw_matrix[src][dst] + \
                                                path_bws[path]
        
        return traffic_class_bw_matrix

    # generate the max min bw allocations for traffic between tor pairs
    # bi_graph: a bipartite graph with path nodes and link nodes
    def get_max_min_fair_bw(self, bi_graph):
        # debug = False

        max_min_fair_path_bws = dict() 
        if bi_graph.size() == 0:
            return max_min_fair_path_bws

        while bi_graph.number_of_edges() > 0: 
            min_link_node = None
            min_bottleneck_bw = sys.maxsize

            link_nodes = [node for node in bi_graph.nodes 
                               if bi_graph.nodes[node]['bipartite'] == 1]
            path_nodes = [node for node in bi_graph.nodes 
                               if bi_graph.nodes[node]['bipartite'] == 0]
                
            for link_node in link_nodes:
                # print(bi_graph[link_node])
                capacity = bi_graph.nodes[link_node]['capacity']
                paths = bi_graph.degree[link_node]
                bottleneck_bw = capacity/paths
                if bottleneck_bw < min_bottleneck_bw:
                    min_link_node = link_node
                    min_bottleneck_bw = bottleneck_bw
            # if debug:
            #     print("link: %s: %s" 
            #           % (min_link_node, bi_graph.nodes[min_link_node]['max_capacity']))
            min_link_paths = list(bi_graph.neighbors(min_link_node))
            bi_graph.remove_node(min_link_node)
            for path_node in min_link_paths:
                max_min_fair_path_bws[path_node] = min_bottleneck_bw
                path_links = list(bi_graph.neighbors(path_node))
                bi_graph.remove_node(path_node)
                for link_node in path_links:
                    bi_graph.nodes[link_node]['capacity'] = \
                            bi_graph.nodes[link_node]['capacity'] - min_bottleneck_bw
                    if bi_graph.degree[link_node] == 0:
                        # if debug:
                        #     print("link: %s: %s" % (link_node, 
                        #           bi_graph.nodes[link_node]['max_capacity'] - \
                        #             bi_graph.nodes[link_node]['capacity']))
                        bi_graph.remove_node(link_node)
                        continue
        
        return max_min_fair_path_bws
    
    # generate a bi-partite graph for waterfilling algorithm computation
    # switch_set: set of switches that are being updated and are not active
    def generate_bi_graph(self, switch_set):
        network = self.network
        traffic_matrix = self.traffic_matrix

        remaining_switch_set = set(network.graph.nodes).difference(set(switch_set))
        subgraph = network.graph.subgraph(remaining_switch_set)
        paths = []

        # get all active paths 
        # assumptions: 
        # 1. traffic at the granularity of (src, dst) tor pairs 
        # 2. traffic is routed on all active shortest paths from a src to a dst tor
        # 3. traffic can be arbitrarily divided
        for src in range(network.num_tor_switches):
            for dst in range(network.num_tor_switches):
                if traffic_matrix[src][dst] > 0:
                    physical_src = network.get_tor_physical_id(src)
                    physical_dst = network.get_tor_physical_id(dst)
                    if nx.has_path(subgraph, physical_src, physical_dst):
                        for path in nx.all_shortest_paths(subgraph, physical_src, physical_dst):
                            paths.append([src, dst, path])

        # make a bi-partite graph (U, V, E)
        # U: set of path nodes
        # V: set of link nodes
        bi_graph = nx.Graph()
        path_counter = 0
        for path in paths: 
            src = path[0]
            dst = path[1]
            pth = path[2]

            path_node = (src, dst, "path_%s" % path_counter)
            bi_graph.add_node(path_node, bipartite=0)
            
            # add a fictitious edge for (src, dst) tor pair at the src 
            # the capacity of this edge is the traffic demand between the (src, dst) pair
            link_node = (src, dst, src)
            bi_graph.add_node(link_node, bipartite=1, 
                                         capacity=traffic_matrix[src][dst],
                                         max_capacity=traffic_matrix[src][dst])
            bi_graph.add_edge(link_node, path_node)

            # add a fictitious edge for (src, dst) tor pair at the dst 
            # the capacity of this edge is the traffic demand between the (src, dst) pair
            link_node = (src, dst, dst)
            bi_graph.add_node(link_node, bipartite=1, 
                                         capacity=traffic_matrix[src][dst],
                                         max_capacity=traffic_matrix[src][dst])
            bi_graph.add_edge(link_node, path_node)

            # add edge between a path node and all link nodes that path traverses
            for j in range(1, len(pth)):
                i = j - 1
                link_node = (pth[i], pth[j], 'link')
                bi_graph.add_node(link_node, bipartite=1, 
                                             capacity=subgraph[pth[i]][pth[j]]['capacity'],
                                             max_capacity=subgraph[pth[i]][pth[j]]['capacity'])
                bi_graph.add_edge(link_node, path_node)

            path_counter = path_counter + 1

        return bi_graph