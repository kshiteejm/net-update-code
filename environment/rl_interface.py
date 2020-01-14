import numpy as np


class RLEnv(object):
    def __init__(self):
        self.reset()

    def step(self, switch_idx):
        # None stands for current step done
        if switch_idx is None:
            reward = self.get_reward(self.intermediate_swiches)
            self.switches_to_update -= self.intermediate_swiches
            self.num_steps -= 1
        else:
            assert not switch_idx in self.intermediate_swiches
            self.intermediate_swiches.add(switch_idx)
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
        self.num_steps = 4
        self.switches_to_update = set()
        self.intermediate_swiches = set()
        # tuple(sorted(down_switch_idx_set)) -> cost
        self.cost_model = {}

    def get_reward(self, down_switch_set):
        # cost model table look up
        return self.cost_model[tuple(sorted(down_switch_set))]

    def get_state(self):
        # takes in topology, switches left to update,
        # intermediate switches (within one mdp step), 
        # traffic matrix, steps left as input,
        # output a graph of state
        pass
