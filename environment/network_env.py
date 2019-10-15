from environment.network_graph import NetworkGraph
from environment.flow_link_graph import FlowLinkGraph

class NetworkEnv():
    def __init__(self, topo_type='FatTree', kwargs={"pods": 4},
                 num_flows=10):
        self.topo_type = topo_type
        self.kwargs = kwargs
        self.num_flows = num_flows
        self.network_graph = NetworkGraph(topo_type, kwargs)
        self.flows = self.network_graph.get_flows(num_flows)
        self.flow_link_graph = FlowLinkGraph(self.network_graph, self.flows)
    
    # def 
