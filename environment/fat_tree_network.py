import networkx as nx
from networkx import bipartite
import random
import os

from itertools import combinations, chain
import numpy as np
import sys, traceback

from graphviz import Graph, Digraph
from IPython.display import display

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

        # init traffic matrix between tor pairs
        self.traffic_matrix = np.zeros((num_tor_switches, num_tor_switches))
        for src in range(num_tor_switches):
            for dst in range(num_tor_switches):
                if src == dst:
                    continue
                self.traffic_matrix[src][dst] = float(random.randint(1000, 1000))
        
        print(np.sum(self.traffic_matrix))

        # init set of switches to be updated in the network
        self.update_switch_set = set()
        for core_logical_id in range(self.num_core_switches):
            self.update_switch_set.add(self.get_core_physical_id(core_logical_id))
        for agg_logical_id in range(self.num_agg_switches):
            self.update_switch_set.add(self.get_agg_physical_id(agg_logical_id))

        self.baseline_bw_matrix, self.baseline_graph = self.generate_baseline_bws()

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

    # generate a powerset except the empty set
    def powerset(self, switch_set):
        s = list(switch_set)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    # generate a bi-partite graph for waterfilling algorithm computation
    # switch_set: set of switches that are being updated and are not active
    def generate_bi_graph(self, switch_set):
        remaining_switch_set = set(self.graph.nodes).difference(set(switch_set))
        subgraph = self.graph.subgraph(remaining_switch_set)
        paths = []

        # get all active paths 
        # assumptions: 
        # 1. traffic at the granularity of (src, dst) tor pairs 
        # 2. traffic is routed on all active shortest paths from a src to a dst tor
        # 3. traffic can be arbitrarily divided
        for src in range(self.num_tor_switches):
            for dst in range(self.num_tor_switches):
                if self.traffic_matrix[src][dst] > 0:
                    physical_src = self.get_tor_physical_id(src)
                    physical_dst = self.get_tor_physical_id(dst)
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
                                         capacity=self.traffic_matrix[src][dst],
                                         max_capacity=self.traffic_matrix[src][dst])
            bi_graph.add_edge(link_node, path_node)

            # add a fictitious edge for (src, dst) tor pair at the dst 
            # the capacity of this edge is the traffic demand between the (src, dst) pair
            link_node = (src, dst, dst)
            bi_graph.add_node(link_node, bipartite=1, 
                                         capacity=self.traffic_matrix[src][dst],
                                         max_capacity=self.traffic_matrix[src][dst])
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

        return bi_graph, paths

    # generate the max min bw allocations for traffic between tor pairs
    # bi_graph: a bipartite graph with path nodes and link nodes
    def get_max_min_bw(self, bi_graph, debug=False):
        fair_bw = dict() 
        if bi_graph.size() == 0:
            return fair_bw

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
                fair_bw[path_node] = min_bottleneck_bw
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
        
        return fair_bw

    # get the cost of a max min bw allocation on a graph with a subset of switches taken down
    # the cost function is a linear cost function at the moment
    def get_cost(self, max_min_bw_matrix, baseline_bw_matrix):
        cost = np.sum(abs(baseline_bw_matrix - max_min_bw_matrix))
        return cost

    # generate baselines with all switches active
    def generate_baseline_bws(self):
        # generate baseline max-min bw allocation with all switches active
        baseline_bi_graph, baseline_paths = self.generate_bi_graph(set())
        baseline_bws = self.get_max_min_bw(baseline_bi_graph)
        baseline_bw_matrix = self.get_traffic_class_bw_matrix(baseline_bws)
        baseline_graph = self.get_updated_graph(baseline_paths, baseline_bws, set(), [])

        print(baseline_bw_matrix)
        print(np.sum(baseline_bw_matrix))
        
        return baseline_bw_matrix, baseline_graph

    # generate all possible one-step update costs
    def generate_costs(self, cost_file_name): 
        cost_file = open(cost_file_name, 'w')
        cost_file.write("cost,down_idx\n")

        for switch_set in self.powerset(self.update_switch_set):
            bi_graph, paths = self.generate_bi_graph(switch_set)
            max_min_bw = self.get_max_min_bw(bi_graph)
            max_min_bw_matrix = self.get_traffic_class_bw_matrix(max_min_bw)
            cost = self.get_cost(max_min_bw_matrix, self.baseline_bw_matrix)
            # cost = self.get_cost(max_min_bw_matrix, self.traffic_matrix)
            
            switch_set_string = ""
            for switch in sorted(switch_set):
                switch_set_string =  switch_set_string + str(switch) + ","
            switch_set_string = switch_set_string[:-1]
            
            cost_file.write("%s,%s\n" % (round(cost, 2), switch_set_string))    
        cost_file.close()

    def get_traffic_class_bw_matrix(self, path_bws):
        traffic_class_bw_matrix = np.zeros((self.num_tor_switches, self.num_tor_switches))
        for path in path_bws:
            src = path[0]
            dst = path[1]
            traffic_class_bw_matrix[src][dst] = traffic_class_bw_matrix[src][dst] + path_bws[path]
        return traffic_class_bw_matrix

    def get_updated_graph(self, paths, path_bws, switch_set, action_seq):
        # print(path_bws)
        graph = self.graph.copy()

        for switch in switch_set:
            graph.nodes[switch]['active'] = False
        
        step = 0
        for seq in action_seq:
            for switch in seq:
                graph.nodes[switch]['step'] = step
            step = step + 1

        path_counter = 0
        for path in paths:
            src = path[0]
            dst = path[1]
            pth = path[2]
            path_node = (src, dst, 'path_%s' % path_counter)
            bw = path_bws[path_node]

            # print("BW: %s: %s" % (path_node, bw))
            for j in range(1, len(pth)):
                i = j - 1
                pth_src = pth[i]
                pth_dst = pth[j]
                graph[pth_src][pth_dst]['used_capacity'] = \
                    graph[pth_src][pth_dst]['used_capacity'] + bw
            
            path_counter = path_counter + 1

        # for edge in graph.edges():
        #     print("%s: %s" % (edge, graph.edges[edge]['used_capacity']))

        return graph
    
    def visualize_graph(self, graph, cost, visual_file):
        core_physical_ids = self.get_core_physical_ids()
        agg_physical_ids = self.get_agg_physical_ids()
        tor_physical_ids = self.get_tor_physical_ids()
        physical_ids = [core_physical_ids, agg_physical_ids, tor_physical_ids]

        # TODO: specify better max cap
        pen_scale = self.link_bw

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
        vg = Digraph('switch_graph', engine='neato')

        # place nodes in the graph
        for n in graph.nodes:
            if graph.nodes[n]['active']:
                if graph.nodes[n]['step'] == -1:  
                    color = 'white'
                else:
                    num = (graph.nodes[n]['step'] + 1) * 17
                    color = 'gray%s' % num
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
            color = 'black'
            if coord_dict[src][1] % 2 == 0: 
                color = 'grey'
            weight = graph.edges[e]['used_capacity'] / pen_scale
            # if (dst, src) not in edge_record:
            #     edge_record.add((src, dst))
            # else:
            #     # check 'used_capacity' is the same
            #     np.isclose(graph.edges[e]['used_capacity'],
            #             graph.edges[(dst, src)]['used_capacity'])
            #     continue
            vg.edge(str(src), str(dst), 
                penwidth=str(graph.edges[e]['used_capacity'] / pen_scale), 
                len=str(graph.edges[e]['used_capacity'] / pen_scale),
                color=color)

        # put a text for cost in an invisible node
        vg.node('Cost: {}'.format(cost),
                pos='0.5, {}!'.format(max(v_dict.keys()) + 0.5),
                color='white')

        vg.render(visual_file, view=False)

    def generate_visualization(self, optimal_action_seq_file, 
                               visual_file, optimal_cost_action_file, num_steps):
        f = open(optimal_action_seq_file, 'r')
        action_seq = []
        for line in f.readlines():
            if (line == '""\n'):
                continue
            switch_set = line[:-1]
            switch_set = switch_set.split(',')
            switch_set = [int(switch) for switch in switch_set]
            switch_set = set(switch_set)
            action_seq.append(switch_set)
        f.close()
        print(action_seq)
        
        total_cost = 0
        cost_action_pairs = []
        for switch_set in action_seq:
            bi_graph, paths = self.generate_bi_graph(switch_set)
            max_min_bw = self.get_max_min_bw(bi_graph, True)
            max_min_bw_matrix = self.get_traffic_class_bw_matrix(max_min_bw)
            # print(max_min_bw_matrix)
            cost = self.get_cost(max_min_bw_matrix, self.baseline_bw_matrix)
            # cost = self.get_cost(max_min_bw_matrix, self.traffic_matrix)
            cost_action_pairs.append((cost, switch_set))
            total_cost = total_cost + cost
        
        self.visualize_graph(self.baseline_graph, round(total_cost, 2), visual_file)

        update_switch_set_left = set(self.update_switch_set)
        num_steps_left = num_steps
        cost_sum = 0
        f = open(optimal_cost_action_file, 'w')
        f.write("%s,%s,%s\n" % (num_steps, round(total_cost,2), str(self.update_switch_set)[:-1][1:]))

        for cost_action_pair in cost_action_pairs:
            num_steps_left = num_steps_left - 1
            cost_sum = cost_sum + cost_action_pair[0]
            update_switch_set_left = update_switch_set_left.difference(cost_action_pair[1])
            remaining_total_cost = total_cost - cost_sum
            str_update_switch_set_left = str(update_switch_set_left)
            if str_update_switch_set_left == 'set()':
                str_update_switch_set_left = ''
                f.write("%s,%s\n" % (num_steps_left, round(remaining_total_cost, 2)))
            else:
                str_update_switch_set_left = str_update_switch_set_left[:-1][1:]
                f.write("%s,%s,%s\n" % (num_steps_left, round(remaining_total_cost, 2), 
                                        str_update_switch_set_left))
        f.close()

    def generate_gcn_dataset(self, optimal_cost_action_file, 
                             save_nodefeats_file, save_adjmats_file, save_cost_file):
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
            num_steps_left = float(triplet[0])
            remaining_total_cost = float(triplet[1])
            update_switch_set_left = set()
            for i in range(2, len(triplet)):
                update_switch_set_left.add(int(triplet[i]))
            gcn_dataset.append((num_steps_left, remaining_total_cost, update_switch_set_left))
        f.close()

        for gcn_datum in gcn_dataset:
            num_steps = gcn_datum[0]
            total_cost = gcn_datum[1]
            update_switch_set = gcn_datum[2]

            # create nodes and register node-indices for each node type
            graph = self.baseline_graph
            q_graph = nx.Graph()
            tc = []
            p = []
            l = []
            s = []
            link_node_dict = dict()
            tc_node_dict = dict()
            node_id = 0
            # initialize tc_node_dict
            for src in range(self.num_tor_switches):
                for dst in range(self.num_tor_switches):
                    if src != dst:
                        tc_node_dict[(src, dst)] = []
            
            # initialize s
            for node in graph.nodes:
                to_update = 0.0
                if node in update_switch_set:
                    to_update = 1.0
                q_graph.add_node(node_id, type='s', id=node, raw_feats=[to_update,num_steps])
                s.append(node_id)
                node_id = node_id + 1
            # initialize l
            for edge in graph.edges:
                q_graph.add_node(node_id, type='l', id=edge, 
                                raw_feats=[graph.edges[edge]['capacity'], 
                                        graph.edges[edge]['used_capacity']])
                q_graph.add_edge(node_id, edge[0])
                q_graph.add_edge(node_id, edge[1])
                l.append(node_id)
                link_node_dict[edge] = node_id
                node_id = node_id + 1
            # initialize p
            for src in range(self.num_tor_switches):
                for dst in range(self.num_tor_switches):
                    if src != dst:
                        physical_src = self.get_tor_physical_id(src)
                        physical_dst = self.get_tor_physical_id(dst)
                        if nx.has_path(graph, physical_src, physical_dst):
                            path_num = 0
                            for path in nx.all_shortest_paths(graph, physical_src, physical_dst):
                                q_graph.add_node(node_id, type='p', id=(src,dst,path_num), 
                                                raw_feats=[0.0,0.0])
                                for j in range(1, len(path)):
                                    i = j - 1
                                    path_src = path[i]
                                    path_dst = path[j]
                                    q_graph.add_edge(node_id, link_node_dict[(path_src, path_dst)])
                                p.append(node_id)
                                tc_node_dict[(src, dst)].append(node_id)
                                node_id = node_id + 1
                                path_num = path_num + 1
            # initialize tc
            for src in range(self.num_tor_switches):
                for dst in range(self.num_tor_switches):
                    if src != dst:
                        q_graph.add_node(node_id, type='tc', id=(src,dst), 
                                            raw_feats=[self.traffic_matrix[src][dst],0.0])
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

            np.save("%s_%s" % (save_nodefeats_file, int(num_steps)), 
                [s_node_features, l_node_features, p_node_features, tc_node_features])
            np.save("%s_%s" % (save_adjmats_file, int(num_steps)), 
                [s_adj_matrix, l_adj_matrix, p_adj_matrix, tc_adj_matrix])
            np.save("%s_%s" % (save_cost_file, int(num_steps)),
                [total_cost])

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("python3 fat_tree_network \
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
    num_steps = 4
    total_cost = 0.0

    random.seed(seed)
    fat_tree_network = FatTreeNetwork(pods=pods)

    if generate_cost_file:
        fat_tree_network.generate_costs("%s/costs_fat_tree_%s_pods_%s.csv" 
                                        % (dataset, pods, seed))

    if generate_action_seq:
        os.system("cd %s; \
                  ./target/debug/rust-dp --num-nodes 20 --num-steps %s \
                  --update-idx 0 1 2 3 4 5 8 9 12 13 16 17 \
                  --cm-path %s/costs_fat_tree_%s_pods_%s.csv \
                  --action-seq-path %s/action_seq_%s_pods_%s.csv" 
                  % (rust_dp, num_steps, dataset, pods, seed, dataset, pods, seed))
    
    optimal_cost_action_file = "%s/opt_cost_actions_%s_pods_%s.csv" % (dataset, pods, seed)
    if generate_visualizations:
        fat_tree_network.generate_visualization(
                            "%s/action_seq_%s_pods_%s.csv" 
                            % (dataset, pods, seed), 
                            "%s/graph_fat_tree_%s_pods_%s" 
                            % (dataset, pods, seed),
                            optimal_cost_action_file, 
                            num_steps)

    save_nodefeats_file =  "%s/nodefeats_fat_tree_%s_pods_%s" % (dataset, pods, seed)
    save_adjmats_file =  "%s/adjmats_fat_tree_%s_pods_%s" % (dataset, pods, seed)
    save_cost_file = "%s/cost_fat_tree_%s_pods_%s" % (dataset, pods, seed)
    fat_tree_network.generate_gcn_dataset(optimal_cost_action_file, save_nodefeats_file, 
                                          save_adjmats_file, save_cost_file)
