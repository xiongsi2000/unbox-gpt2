import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))