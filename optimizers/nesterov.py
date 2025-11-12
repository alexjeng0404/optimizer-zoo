import torch
class Nesterov:
    # learning notes
    """
    ref : https://medium.com/@piyushkashyap045/understanding-nesterov-accelerated-gradient-nag-340c53d64597
    Nesterov Accelerated Gradient (NAG):
    Use momentum and NAG(look ahead gradient) to accelerate the update
    """

    def __init__(self, params, lr=0.01, beta=0.9):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                prev_v = self.v[i].clone()
                self.v[i] = self.beta * self.v[i] + (1 - self.beta) * p.grad
                p -= self.lr * ((1 + self.beta) * self.v[i] - self.beta * prev_v)

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.zero_()