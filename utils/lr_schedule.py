class LinearLearnRateScheduler(object):
    def __init__(self, optimizers, lr_init, lr_final, lr_epochs):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_epochs = lr_epochs
        self.optimizers = optimizers

    def set_rate(self, num_epoch):
        # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
        lr = self.lr_final
        if num_epoch < self.lr_epochs:
            lr = self.lr_init + ((self.lr_final - self.lr_init) * num_epoch)/self.lr_epochs
        for optimizer in self.optimizers:
            state_dict = optimizer.state_dict()
            for param_group in state_dict['param_groups']:
                param_group['lr'] = lr
            optimizer.load_state_dict(state_dict)
        return lr