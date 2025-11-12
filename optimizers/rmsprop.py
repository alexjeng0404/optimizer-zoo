import torch

class Rmsprop:
    # learning notes
    """
    RMSProp: depends on the recent mean-square of the gradient.
    The closer the value beta is to 1, the longer the memory of the past is
    """

    def __init__(self, params, lr=0.01, beta=0.9, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                self.v[i] = self.beta * self.v[i] + (1 - self.beta) * g * g
                p -= self.lr * g / (torch.sqrt(self.v[i]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.zero_()