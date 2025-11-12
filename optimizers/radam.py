import torch
import math

class RAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
                
                beta2_t = self.beta2 ** self.t
                N_sma_max = 2 / (1 - self.beta2) - 1
                N_sma = N_sma_max - 2 * self.t * beta2_t / (1 - beta2_t)
                
                if N_sma > 5:
                    rect = math.sqrt(((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * 
                                    (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)))
                    step_size = self.lr * rect / (1 - self.beta1 ** self.t)
                    p -= step_size * self.m[i] / (torch.sqrt(self.v[i]) + self.eps)
                else:
                    step_size = self.lr / (1 - self.beta1 ** self.t)
                    p -= step_size * self.m[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()