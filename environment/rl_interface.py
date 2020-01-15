import numpy as np
from environment.dcn_environment import DCNEnvironment


class RLEnv(object):
    def __init__(self):
        self.reset()

    def step(self, switch_idx):
        assert(switch_idx is None or \
               switch_idx in self.switches_to_update)

        # None stands for current step done
        if switch_idx is None:
            reward = self.get_reward(self.intermediate_switches)
            self.switches_to_update -= self.intermediate_switches
            self.num_steps -= 1
        else:
            assert not switch_idx in self.intermediate_switches
            self.intermediate_switches.add(switch_idx)
            reward = 0

        state = self.get_state()

        if self.num_steps == 1:
            done = True
            # there is no more action to take
            # we have to take down the remaining switches
            reward += self.get_reward(self.switches_to_update)
        else:
            done = False

        return state, reward, done

    def reset(self):
        # update a new environment
        # potentially new topology, new set of
        # switches to update, new traffic matrix
        self.dcn_environment = DCNEnvironment(pods=4, link_bw=10000.0, max_num_steps=4)
        self.num_steps = self.dcn_environment.get_max_num_steps()
        self.switches_to_update = self.dcn_environment.get_update_switch_set()
        self.intermediate_switches = set()
        # tuple(sorted(down_switch_idx_set)) -> cost
        # self.cost_model = self.dcn_environment.get_cost_model()
        state = self.get_state()
        return state

    def get_reward(self, down_switch_set):
        # cost model table look up
        # return self.cost_model[tuple(sorted(down_switch_set))]
        updated_traffic_matrix = self.dcn_environment\
                                     .max_min_fair_bw_calculator\
                                     .get_traffic_class_fair_bw_matrix(down_switch_set)
        cost = self.dcn_environment.cost_function.get_cost(updated_traffic_matrix)
        return -cost

    def get_state(self):
        # takes in topology, switches left to update,
        # intermediate switches (within one mdp step), 
        # traffic matrix, steps left as input,
        # output a graph of state
        switch_mask = np.zeros(self.dcn_environment.get_total_switches() + 1)
        switch_mask[:-1] = 1
        for switch_id in (self.switches_to_update - self.intermediate_switches):
            switch_mask[switch_id] = 1.0
        node_feats, adj_mats = self.dcn_environment.get_state(
            self.switches_to_update, self.intermediate_switches, self.num_steps)
        return node_feats, adj_mats, switch_mask
