import torch
import torch.nn as nn

from .feed_forward_network import FeedForwardNetwork
from .multi_head_attention import MultiHeadSelfAttention
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_p_drop, residual_p_drop):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_p_drop)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.dropout = nn.Dropout(residual_p_drop)
    
    @property
    def attention(self):
        """Alias for attn to support both naming conventions"""
        return self.attn

    def forward(self, x):
        x0 = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x0 + x

        x0 = x
        x = self.ln2(x)
        x = self.ffn(x)
        return x0 + x