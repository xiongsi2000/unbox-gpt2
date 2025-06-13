import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        dim = x.size(-1)
        rms = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) / dim + self.epsilon)
        rms_norm = x / rms * self.weight.view(1, -1)
        return rms_norm