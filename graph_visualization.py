from graphviz import Graph
from IPython.display import display
import networkx as nx
import numpy as np


graph_file = './data/graph_5,4.yaml'

# read graph from file
g = nx.read_yaml(graph_file)

# TODO: read the logical index
logic_idx = [[0, 1, 2, 3],
             [4, 5, 8, 9, 12, 13, 16, 17],
             [6, 7, 10, 11, 14, 15, 18, 19]]

# TODO: read these from file too
cost = 38330.0
switch_down = [4, 5]

# TODO: automate vertical coordinate
v_dict = {0: 2, 1: 1, 2: 0}

# TODO: specify better max cap
pen_scale = 5000

# generate coordinate for nodes
h_max = max(len(l) for l in logic_idx)
coord_dict = {}
for i in range(len(logic_idx)):
    h = len(logic_idx[i])
    offset = int((h_max - h) / 2)
    for j in range(len(logic_idx[i])):
        phys_idx = logic_idx[i][j]
        coord_dict[phys_idx] = \
        (j + offset , v_dict[i])  # looks better

# graph visualization object (pos needs neato)
vg = Graph('switch_graph', engine='neato')

# place nodes in the graph
for n in g.nodes:
    if g.nodes[n]['active']:
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
for e in g.edges:
    src = e[0]
    dst = e[1]
    if (dst, src) not in edge_record:
        edge_record.add((src, dst))
    else:
        # check 'used_capacity' is the same
        np.isclose(g.edges[e]['used_capacity'],
                   g.edges[(dst, src)]['used_capacity'])
        continue
    vg.edge(str(src), str(dst), penwidth=str(
        g.edges[e]['used_capacity'] / pen_scale))

# put a text for cost in an invisible node
vg.node('Cost: {}'.format(cost),
        pos='0.5, {}!'.format(max(v_dict.keys()) + 0.5),
        color='white')

vg.render('./data/visualization', view=True)