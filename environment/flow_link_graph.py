import networkx as nx
import random

class FlowLinkGraph:
    def __init__(self, network_graph, flows):
        self.graph = self.create_graph(network_graph, flows)

    def create_graph(self, network_graph, flows):
        graph = nx.Graph()
        num_nodes = len(graph.edges) + len(flows)
        graph.add_nodes_from(range(num_nodes))
        