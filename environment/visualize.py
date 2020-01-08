class Visualize:
    def __init__(self):
        network = 1

    # def get_updated_graph(self, paths, path_bws, switch_set, action_seq):
    #     # print(path_bws)
    #     graph = self.graph.copy()

    #     for switch in switch_set:
    #         graph.nodes[switch]['active'] = False
        
    #     step = 0
    #     for seq in action_seq:
    #         for switch in seq:
    #             graph.nodes[switch]['step'] = step
    #         step = step + 1

    #     path_counter = 0
    #     for path in paths:
    #         src = path[0]
    #         dst = path[1]
    #         pth = path[2]
    #         path_node = (src, dst, 'path_%s' % path_counter)
    #         bw = path_bws[path_node]

    #         # print("BW: %s: %s" % (path_node, bw))
    #         for j in range(1, len(pth)):
    #             i = j - 1
    #             pth_src = pth[i]
    #             pth_dst = pth[j]
    #             graph[pth_src][pth_dst]['used_capacity'] = \
    #                 graph[pth_src][pth_dst]['used_capacity'] + bw
            
    #         path_counter = path_counter + 1

    #     # for edge in graph.edges():
    #     #     print("%s: %s" % (edge, graph.edges[edge]['used_capacity']))

    #     return graph
    
    # def visualize_graph(self, graph, cost, visual_file):
    #     core_physical_ids = self.get_core_physical_ids()
    #     agg_physical_ids = self.get_agg_physical_ids()
    #     tor_physical_ids = self.get_tor_physical_ids()
    #     physical_ids = [core_physical_ids, agg_physical_ids, tor_physical_ids]

    #     # TODO: specify better max cap
    #     pen_scale = self.link_bw

    #     # TODO: automate vertical coordinate
    #     v_dict = {0: 2, 1: 1, 2: 0}

    #     # generate coordinate for nodes
    #     h_max = max(len(l) for l in physical_ids)
    #     coord_dict = {}
    #     for i in range(len(physical_ids)):
    #         h = len(physical_ids[i])
    #         offset = int((h_max - h) / 2)
    #         for j in range(len(physical_ids[i])):
    #             physical_id = physical_ids[i][j]
    #             coord_dict[physical_id] = \
    #             (j + offset , v_dict[i])  # looks better
        
    #     # graph visualization object (pos needs neato)
    #     vg = Digraph('switch_graph', engine='neato')

    #     # place nodes in the graph
    #     for n in graph.nodes:
    #         if graph.nodes[n]['active']:
    #             if graph.nodes[n]['step'] == -1:  
    #                 color = 'white'
    #             else:
    #                 num = (graph.nodes[n]['step'] + 1) * 17
    #                 color = 'gray%s' % num
    #         else:
    #             color = 'dimgray'
    #         vg.node(str(n), pos='{}, {}!'.format(
    #                 coord_dict[n][0], coord_dict[n][1]),
    #                 shape='circle', fillcolor=color,
    #                 style='filled')

    #     # TODO: (check) prevent replotting the same edge
    #     edge_record = set()

    #     # visualize edges
    #     for e in graph.edges:
    #         src = e[0]
    #         dst = e[1]
    #         color = 'black'
    #         if coord_dict[src][1] % 2 == 0: 
    #             color = 'grey'
    #         weight = graph.edges[e]['used_capacity'] / pen_scale
    #         # if (dst, src) not in edge_record:
    #         #     edge_record.add((src, dst))
    #         # else:
    #         #     # check 'used_capacity' is the same
    #         #     np.isclose(graph.edges[e]['used_capacity'],
    #         #             graph.edges[(dst, src)]['used_capacity'])
    #         #     continue
    #         vg.edge(str(src), str(dst), 
    #             penwidth=str(graph.edges[e]['used_capacity'] / pen_scale), 
    #             len=str(graph.edges[e]['used_capacity'] / pen_scale),
    #             color=color)

    #     # put a text for cost in an invisible node
    #     vg.node('Cost: {}'.format(cost),
    #             pos='0.5, {}!'.format(max(v_dict.keys()) + 0.5),
    #             color='white')

    #     vg.render(visual_file, view=False)

    # def generate_visualization(self, optimal_action_seq_file, 
    #                            visual_file, optimal_cost_action_file, num_steps):
    #     f = open(optimal_action_seq_file, 'r')
    #     action_seq = []
    #     for line in f.readlines():
    #         if (line == '""\n'):
    #             continue
    #         switch_set = line[:-1]
    #         switch_set = switch_set.split(',')
    #         switch_set = [int(switch) for switch in switch_set]
    #         switch_set = set(switch_set)
    #         action_seq.append(switch_set)
    #     f.close()
    #     print(action_seq)
        
    #     total_cost = 0
    #     cost_action_pairs = []
    #     for switch_set in action_seq:
    #         bi_graph, paths = self.generate_bi_graph(switch_set)
    #         max_min_bw = self.get_max_min_bw(bi_graph, True)
    #         max_min_bw_matrix = self.get_traffic_class_bw_matrix(max_min_bw)
    #         # print(max_min_bw_matrix)
    #         cost = self.get_cost(max_min_bw_matrix, self.baseline_bw_matrix)
    #         # cost = self.get_cost(max_min_bw_matrix, self.traffic_matrix)
    #         cost_action_pairs.append((cost, switch_set))
    #         total_cost = total_cost + cost
        
    #     self.visualize_graph(self.baseline_graph, round(total_cost, 2), visual_file)

    #     update_switch_set_left = set(self.update_switch_set)
    #     num_steps_left = num_steps
    #     cost_sum = 0
    #     f = open(optimal_cost_action_file, 'w')
    #     f.write("%s,%s,%s\n" % (num_steps, round(total_cost,2), str(self.update_switch_set)[:-1][1:]))

    #     for cost_action_pair in cost_action_pairs:
    #         num_steps_left = num_steps_left - 1
    #         cost_sum = cost_sum + cost_action_pair[0]
    #         update_switch_set_left = update_switch_set_left.difference(cost_action_pair[1])
    #         remaining_total_cost = total_cost - cost_sum
    #         str_update_switch_set_left = str(update_switch_set_left)
    #         if str_update_switch_set_left == 'set()':
    #             str_update_switch_set_left = ''
    #             f.write("%s,%s\n" % (num_steps_left, round(remaining_total_cost, 2)))
    #         else:
    #             str_update_switch_set_left = str_update_switch_set_left[:-1][1:]
    #             f.write("%s,%s,%s\n" % (num_steps_left, round(remaining_total_cost, 2), 
    #                                     str_update_switch_set_left))
    #     f.close()
