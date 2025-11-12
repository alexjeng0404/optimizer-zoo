import torch

class CyclicalLR:
    def __init__(self, base_lr, max_lr, step_size):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iter = 0

    def get_lr(self):
        cycle = torch.floor(1 + self.iter / (2 * self.step_size))
        x = torch.abs(self.iter / self.step_size - 2 * cycle + 1)
        scale = torch.max(torch.tensor(0.0), 1 - x)
        lr = self.base_lr + (self.max_lr - self.base_lr) * scale
        self.iter += 1
        return lr.item()