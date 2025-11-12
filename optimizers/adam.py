import numpy as np
import torch

class Adam:
    """
    Adam optimizer
    Combines Momentum and RMSProp with bias correction.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
    m_hat = m_t / (1 - beta1 ** t)
    v_hat = v_t / (1 - beta2 ** t)
    theta_t+1 = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * g * g
                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.zero_()