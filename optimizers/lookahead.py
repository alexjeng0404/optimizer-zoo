import torch

class Lookahead:
    """
    Lookahead Optimizer
    Ref:
      - Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back"
        (NeurIPS 2019)
      - https://arxiv.org/abs/1907.08610
    """
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k              # synchronization interval
        self.alpha = alpha      # slow weight update factor
        self.step_counter = 0

        if hasattr(base_optimizer, 'param_groups'):
            param_groups = base_optimizer.param_groups
        else:
            param_groups = [{'params': base_optimizer.params}]

        # initialize slow weights
        self.slow_params = []
        for group in param_groups:
            slow_group = []
            for p in group['params']:
                slow_p = p.clone().detach()
                slow_group.append(slow_p)
            self.slow_params.append(slow_group)

        # safe param_groups
        self._param_groups = param_groups

    def step(self):
        # perform one inner optimizer step
        self.base_optimizer.step()
        self.step_counter += 1

        # every k steps, update slow weights and sync fast weights
        if self.step_counter % self.k == 0:
            for group_idx, group in enumerate(self._param_groups):
                for param_idx, p in enumerate(group['params']):
                    slow_p = self.slow_params[group_idx][param_idx]
                    
                    # Lookahead update: slow = slow + alpha * (fast - slow)
                    new_slow = slow_p + self.alpha * (p.data - slow_p)
                    
                    # update slow weights
                    self.slow_params[group_idx][param_idx].copy_(new_slow)
                    
                    # synchronize fast weights to new slow weights
                    p.data.copy_(new_slow)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def param_groups(self):
        """assign param_groups"""
        if hasattr(self.base_optimizer, 'param_groups'):
            return self.base_optimizer.param_groups
        else:
            return self._param_groups