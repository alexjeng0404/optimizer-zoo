import torch

class AdaBelief:
    '''
    Instead of g^2, we use (g - m_t)^2 to measure the uncertainty of the gradient, resulting in more stable convergence.
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):  # add weight_decay
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.s = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                if self.weight_decay != 0:
                    g = g.add(p, alpha=self.weight_decay)

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                # AdaBelief key difference (g - m_t)^2, instead of g^2
                self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * (g - self.m[i]).square()

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                s_hat = self.s[i] / (1 - self.beta2 ** self.t)

                p -= self.lr * m_hat / (torch.sqrt(s_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()