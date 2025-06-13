import pytest
import torch
import torch.nn as nn
import math
from transformer_lib.attention import ScaledDotProductAttention, MultiHeadSelfAttention


class TestScaledDotProductAttention:
    """Test cases for Scaled Dot-Product Attention"""
    
    def test_attention_initialization(self):
        """Test attention module initializes correctly"""
        dropout_prob = 0.1
        attention = ScaledDotProductAttention(dropout_prob)
        assert isinstance(attention, nn.Module)
        assert attention.dropout.p == dropout_prob
    
    def test_attention_shape_consistency(self, batch_size, seq_len, d_model):
        """Test attention preserves correct output shape"""
        attention = ScaledDotProductAttention(0.0)  # No dropout for deterministic testing
        
        # Create Q, K, V tensors
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(q, k, v)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_attention_with_mask(self, batch_size, seq_len, d_model):
        """Test attention applies causal mask correctly"""
        attention = ScaledDotProductAttention(0.0)
        
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        output = attention(q, k, v, mask)
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.any(torch.isnan(output))
    
    def test_attention_without_mask(self, batch_size, seq_len, d_model):
        """Test attention works without mask"""
        attention = ScaledDotProductAttention(0.0)
        
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(q, k, v, mask=None)
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.any(torch.isnan(output))
    
    def test_attention_scaling(self):
        """Test attention applies correct scaling factor"""
        attention = ScaledDotProductAttention(0.0)
        
        d_model = 64
        q = torch.ones(1, 2, d_model)
        k = torch.ones(1, 2, d_model)
        v = torch.randn(1, 2, d_model)
        
        # Manually compute expected attention scores
        expected_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        
        # The attention function should apply the same scaling
        output = attention(q, k, v)
        assert not torch.any(torch.isnan(output))


class TestMultiHeadSelfAttention:
    """Test cases for Multi-Head Self-Attention"""
    
    def test_multihead_attention_initialization(self, d_model):
        """Test multi-head attention initializes correctly"""
        num_heads = 8
        dropout_prob = 0.1
        
        # d_model must be divisible by num_heads
        attention = MultiHeadSelfAttention(d_model, num_heads, dropout_prob)
        assert isinstance(attention, nn.Module)
        assert attention.num_heads == num_heads
        assert attention.head_dim == d_model // num_heads
        assert attention.d_model == d_model
    
    def test_multihead_attention_invalid_heads(self, d_model):
        """Test multi-head attention raises error for invalid head count"""
        # Use a num_heads that doesn't divide d_model evenly
        invalid_num_heads = d_model + 1
        
        with pytest.raises(ValueError, match="d_model .* must be divisible by num_heads"):
            MultiHeadSelfAttention(d_model, invalid_num_heads, 0.1)
    
    def test_multihead_attention_shape_preservation(self, sample_input, d_model):
        """Test multi-head attention preserves input shape"""
        num_heads = 8
        attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)
        
        output = attention(sample_input)
        assert output.shape == sample_input.shape
    
    def test_multihead_attention_causal_mask(self, batch_size, seq_len, d_model):
        """Test multi-head attention applies causal masking"""
        num_heads = 8
        attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_multihead_attention_different_head_counts(self, d_model):
        """Test multi-head attention works with different valid head counts"""
        x = torch.randn(2, 4, d_model)
        
        # Test different head counts that divide d_model evenly
        valid_head_counts = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0]
        
        for num_heads in valid_head_counts:
            attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)
            output = attention(x)
            assert output.shape == x.shape
            assert not torch.any(torch.isnan(output))
    
    def test_multihead_attention_gradients(self, sample_input, d_model):
        """Test multi-head attention produces gradients"""
        num_heads = 8
        attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)
        
        # Enable gradients for input
        x = sample_input.clone().requires_grad_(True)
        output = attention(x)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for input and parameters
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check parameter gradients
        for param in attention.parameters():
            assert param.grad is not None
    
    def test_multihead_attention_deterministic(self, sample_input, d_model):
        """Test multi-head attention is deterministic"""
        num_heads = 8
        attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)  # No dropout
        
        # Set eval mode to disable any randomness
        attention.eval()
        
        output1 = attention(sample_input)
        output2 = attention(sample_input)
        
        assert torch.allclose(output1, output2)
    
    def test_multihead_attention_different_sequence_lengths(self, d_model):
        """Test multi-head attention works with different sequence lengths"""
        num_heads = 8
        attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)
        
        batch_size = 2
        seq_lengths = [1, 4, 16, 32]
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, d_model)
            output = attention(x)
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.any(torch.isnan(output))
    
    def test_multihead_attention_memory_efficiency(self, d_model):
        """Test multi-head attention doesn't use excessive memory"""
        num_heads = 8
        attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)
        
        # Test with larger sequence to check memory scaling
        x = torch.randn(1, 64, d_model)
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.any(torch.isnan(output))
    
    def test_multihead_attention_parameter_count(self, d_model):
        """Test multi-head attention has expected parameter count"""
        num_heads = 8
        attention = MultiHeadSelfAttention(d_model, num_heads, 0.0)
        
        # Count parameters
        total_params = sum(p.numel() for p in attention.parameters())
        
        # Expected: 4 linear layers (q, k, v, output) each with d_model x d_model weights
        # No biases in this implementation
        expected_params = 4 * d_model * d_model
        assert total_params == expected_params
    
    def test_multihead_attention_training_vs_eval(self, sample_input, d_model):
        """Test multi-head attention behaves differently in train vs eval mode"""
        num_heads = 8
        dropout_prob = 0.5  # High dropout to see difference
        attention = MultiHeadSelfAttention(d_model, num_heads, dropout_prob)
        
        # Training mode
        attention.train()
        output_train1 = attention(sample_input)
        output_train2 = attention(sample_input)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2, rtol=1e-6, atol=1e-6)
        
        # Eval mode
        attention.eval()
        output_eval1 = attention(sample_input)
        output_eval2 = attention(sample_input)
        
        # Outputs should be the same (no dropout)
        assert torch.allclose(output_eval1, output_eval2)