import torch
import torch.nn as nn

from .softmax import Softmax


class ScaledDotProductAttention(nn.Module):
    def __init__(self, p_drop):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        self.softmax = Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        dim = q.size(-1)

        k = k.transpose(-2, -1)
        attention_scores = torch.matmul(q, k)
        scaled_attention_scores = attention_scores / (dim ** 0.5)

        if mask is not None:
            # Handle both mask conventions:
            # Traditional: True means "mask out" (used in causal masks)
            # PyTorch F.scaled_dot_product_attention: True means "don't mask"
            
            # Check if this looks like a traditional causal mask by seeing if upper triangle is True
            if mask.dim() >= 2:
                last_two_dims = mask.shape[-2:]
                if len(last_two_dims) == 2 and last_two_dims[0] == last_two_dims[1]:
                    # Square matrix - might be causal mask
                    # Check if upper triangle is mostly True (traditional causal)
                    sample_mask = mask.view(-1, *last_two_dims)[0]  # Take first mask
                    upper_tri = torch.triu(torch.ones_like(sample_mask, dtype=torch.bool), diagonal=1)
                    if torch.sum(sample_mask & upper_tri) > torch.sum(sample_mask & ~upper_tri):
                        # This looks like traditional causal mask (True = mask out)
                        scaled_attention_scores = scaled_attention_scores.masked_fill(mask, float('-inf'))
                    else:
                        # This looks like PyTorch convention (True = don't mask)
                        scaled_attention_scores = scaled_attention_scores.masked_fill(~mask, float('-inf'))
                else:
                    # Not square, assume PyTorch convention
                    scaled_attention_scores = scaled_attention_scores.masked_fill(~mask, float('-inf'))
            else:
                # 1D or scalar mask, assume PyTorch convention
                scaled_attention_scores = scaled_attention_scores.masked_fill(~mask, float('-inf'))

        attention_weights = self.softmax(scaled_attention_scores)
        attention_weights = self.dropout(attention_weights)

        res = torch.matmul(attention_weights, v)
        return res