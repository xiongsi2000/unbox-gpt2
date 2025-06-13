import torch
import torch.nn as nn

from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_p_drop):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop_out = nn.Dropout(attn_p_drop)
        
        self.attention = ScaledDotProductAttention(attn_p_drop)

    def _generate_causal_mask(self, B, H, Seq_L, device):
        # Create causal mask where True means "don't mask" (keep the value)
        # Lower triangular matrix (including diagonal) should be True
        matrix = torch.tril(torch.ones(Seq_L, Seq_L)).bool().to(device)
        matrix = matrix.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        return matrix

    def forward(self, x):
        device = x.device
        batch_size, seq_len, _ = x.shape

        mask = self._generate_causal_mask(batch_size, self.num_heads, seq_len, device)

        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        q_proj = q_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        x = self.attention(q_proj, k_proj, v_proj, mask)

        x = x.transpose(1, 2).contiguous()
        attention_weights = x.view(batch_size, -1, self.d_model)

        x = self.output_proj(attention_weights)
        return x