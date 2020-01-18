import heapq
import numpy as np
from param import config


class TrajStore(object):
    def __init__(self):
        self.pq = []
        self.curr_q = []
        self.r = 0

    def add(self, node_feats, next_node_feats, adj_mats, next_adj_mats,
            switch_mask, switch_a, reward, pi, done):
        
        self.curr_q.append((node_feats, next_node_feats,
                            adj_mats, next_adj_mats, switch_mask,
                            switch_a, reward, pi, done))
        self.r += reward

        if done:
            heapq.heappush(self.pq, (self.r, self.curr_q))
            if len(self.pq) > config.heapq_size:
                heapq.heappop(self.pg)
            self.curr_q = []
            self.r = 0

    def get(self):
        # randomly sample a trajectory from pq
        idx = np.random.randint(len(self.pq))
        return self.pq[idx][1]
