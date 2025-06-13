#!/usr/bin/env python3
"""
Generate reference fixtures for testing
"""
import torch
import math
from pathlib import Path

# Set seed for reproducible fixtures
torch.manual_seed(42)

FIXTURES_PATH = Path(__file__).parent / "fixtures"
FIXTURES_PATH.mkdir(exist_ok=True)

def generate_test_fixtures():
    """Generate reference test data and expected outputs"""
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_ff = 16
    num_heads = 2
    vocab_size = 20
    
    print("Generating test fixtures...")
    
    # 1. Generate GELU test cases
    gelu_input = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
    gelu_expected = 0.5 * gelu_input * (1 + torch.erf(gelu_input / math.sqrt(2.0)))
    torch.save(gelu_input, FIXTURES_PATH / "gelu_input.pt")
    torch.save(gelu_expected, FIXTURES_PATH / "gelu_expected.pt")
    
    # 2. Generate RMSNorm test cases
    rmsnorm_input = torch.randn(batch_size, seq_len, d_model)
    # Compute expected RMSNorm output manually
    dim = rmsnorm_input.size(-1)
    rms = torch.sqrt((rmsnorm_input ** 2).sum(dim=-1, keepdim=True) / dim + 1e-5)
    rmsnorm_weight = torch.ones(d_model)
    rmsnorm_expected = rmsnorm_input / rms * rmsnorm_weight.view(1, -1)
    
    torch.save(rmsnorm_input, FIXTURES_PATH / "rmsnorm_input.pt")
    torch.save(rmsnorm_weight, FIXTURES_PATH / "rmsnorm_weight.pt")
    torch.save(rmsnorm_expected, FIXTURES_PATH / "rmsnorm_expected.pt")
    
    # 3. Generate Feedforward test cases
    ffn_input = torch.randn(batch_size, seq_len, d_model)
    ffn_w1_weight = torch.randn(d_ff, d_model)
    ffn_w2_weight = torch.randn(d_model, d_ff)
    
    # Manual FFN computation
    h = torch.matmul(ffn_input, ffn_w1_weight.T)
    h_gelu = 0.5 * h * (1 + torch.erf(h / math.sqrt(2.0)))
    ffn_expected = torch.matmul(h_gelu, ffn_w2_weight.T)
    
    torch.save(ffn_input, FIXTURES_PATH / "ffn_input.pt")
    torch.save(ffn_w1_weight, FIXTURES_PATH / "ffn_w1_weight.pt")
    torch.save(ffn_w2_weight, FIXTURES_PATH / "ffn_w2_weight.pt")
    torch.save(ffn_expected, FIXTURES_PATH / "ffn_expected.pt")
    
    # 4. Generate Softmax test cases
    softmax_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # Manual stable softmax
    x_max, _ = torch.max(softmax_input, dim=-1, keepdim=True)
    x_stable = softmax_input - x_max
    exp_x = torch.exp(x_stable)
    softmax_expected = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    
    torch.save(softmax_input, FIXTURES_PATH / "softmax_input.pt")
    torch.save(softmax_expected, FIXTURES_PATH / "softmax_expected.pt")
    
    # 5. Generate Attention test cases
    torch.manual_seed(42)
    attention_q = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads)
    attention_k = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads)
    attention_v = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads)
    
    # Manual attention computation
    head_dim = d_model // num_heads
    k_t = attention_k.transpose(-2, -1)
    scores = torch.matmul(attention_q, k_t) / math.sqrt(head_dim)
    
    # Apply causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax
    scores_max, _ = torch.max(scores, dim=-1, keepdim=True)
    scores_stable = scores - scores_max
    exp_scores = torch.exp(scores_stable)
    attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
    
    attention_expected = torch.matmul(attention_weights, attention_v)
    
    torch.save(attention_q, FIXTURES_PATH / "attention_q.pt")
    torch.save(attention_k, FIXTURES_PATH / "attention_k.pt")
    torch.save(attention_v, FIXTURES_PATH / "attention_v.pt")
    torch.save(mask, FIXTURES_PATH / "attention_mask.pt")
    torch.save(attention_expected, FIXTURES_PATH / "attention_expected.pt")
    
    # Save test configuration
    config = {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'd_model': d_model,
        'd_ff': d_ff,
        'num_heads': num_heads,
        'vocab_size': vocab_size
    }
    torch.save(config, FIXTURES_PATH / "test_config.pt")
    
    print(f"✅ Fixtures generated successfully in {FIXTURES_PATH}")
    print(f"Generated {len(list(FIXTURES_PATH.glob('*.pt')))} fixture files")

if __name__ == "__main__":
    generate_test_fixtures()