import torch
import torch.nn as nn
from .gelu import GELU


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gelu = GELU()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        return x