import torch

class SGD:
    # learning notes
    """
    Stochastic Gradient Decent (SGD):
    W(t) = W(t-1) - lr * ∇L
    W is weight parameter, L is loss function, lr is learning rate,
    ∇L is the gradient of L, representing the direction we want to update.
    """
    
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.zero_()