import torch


l2_loss = torch.nn.MSELoss(reduction='mean')

def value_train(v_opt, values, returns):
    v_loss = l2_loss(values, returns)
    v_opt.zero_grad()
    v_loss.backward()
    v_opt.step()
    return v_loss.item()
