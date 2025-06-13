import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        x_stable = x - x_max
        
        exp_x = torch.exp(x_stable)
        softmax = exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)
        
        return softmax