import torch

class Momentum:
    # learning notes
    """
    Momentum : 
    It's like the momentum concept in physics, we update the weight by its momentum (velocity).
    V(t) = β * V(t-1) + lr * ∇L
    W(t) = W(t-1) - V(t)
    V(t) is related to the velocity last time(V(t-1)), β is like a dmaping factor when V(t)
    and V(t-1) are in the opposite direction.
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
                self.v[i] = self.beta * self.v[i] + (1 - self.beta) * p.grad
                p -= self.lr * self.v[i]

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.zero_()