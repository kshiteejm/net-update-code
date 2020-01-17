import torch
import torch.nn.functional as F


def policy_gradient(pi_nn, pi_opt, packed_states, actions,
                    adv, entropy_factor):

    # feed forward policy network
    log_pi, pi, masked_pi = pi_nn(*packed_states)

    # pick based on actions
    log_pi_acts = log_pi.gather(1, actions)

    # entropy loss
    entropy = (log_pi * pi).sum(dim=-1).mean()

    # TODO: importance sampling

    # policy loss
    pg_loss = - (log_pi_acts * adv).mean()

    # loss for backpropagation
    loss = pg_loss + entropy_factor * entropy

    # gradient descent
    pi_opt.zero_grad()
    loss.backward()
    pi_opt.step()

    return pg_loss.item(), entropy.item()