import torch

class Adagrad:
    # learning notes
    """
    AdaGrad: learning rate can now be adjusted. Based on the past rms of the gradient
    Cons: will end up being slow
    """
    
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.G = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                self.G[i] += g * g
                p -= self.lr * g / (torch.sqrt(self.G[i]) + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.zero_()