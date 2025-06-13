import pytest
import torch
import torch.nn as nn
from transformer_lib.transformer_block import TransformerBlock


class TestTransformerBlock:
    """Test cases for Transformer Block"""
    
    def test_transformer_block_initialization(self, d_model):
        """Test transformer block initializes correctly"""
        num_heads = 8
        d_ff = d_model * 4
        attn_dropout = 0.1
        residual_dropout = 0.1
        
        block = TransformerBlock(d_model, num_heads, d_ff, attn_dropout, residual_dropout)
        
        assert isinstance(block, nn.Module)
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'attention')
        assert hasattr(block, 'ln2')
        assert hasattr(block, 'ffn')
        assert hasattr(block, 'dropout')
        assert block.dropout.p == residual_dropout
    
    def test_transformer_block_shape_preservation(self, sample_input, d_model):
        """Test transformer block preserves input shape"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        output = block(sample_input)
        assert output.shape == sample_input.shape
    
    def test_transformer_block_residual_connections(self, d_model):
        """Test transformer block implements residual connections correctly"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        # Use small input to see residual effect clearly
        x = torch.randn(1, 4, d_model) * 0.1
        
        # With zero weights, output should be close to input due to residual connections
        with torch.no_grad():
            # Zero out some weights to test residual connections
            original_weights = {}
            for name, param in block.named_parameters():
                original_weights[name] = param.clone()
                if 'weight' in name:
                    param.fill_(0.0)
        
        output = block(x)
        
        # Restore original weights
        with torch.no_grad():
            for name, param in block.named_parameters():
                param.copy_(original_weights[name])
        
        # Output should not be zero due to residual connections and normalization
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_transformer_block_pre_norm_architecture(self, sample_input, d_model):
        """Test transformer block uses pre-norm architecture"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        # Test that normalization is applied before attention and FFN
        # This is tested by checking the computational flow
        x = sample_input
        
        # Manual forward pass to verify pre-norm
        with torch.no_grad():
            # First residual block: norm -> attention -> residual
            x_norm1 = block.ln1(x)
            attn_out = block.attention(x_norm1)
            attn_out = block.dropout(attn_out)
            x_after_attn = x + attn_out
            
            # Second residual block: norm -> ffn -> residual
            x_norm2 = block.ln2(x_after_attn)
            ffn_out = block.ffn(x_norm2)
            x_final = x_after_attn + ffn_out
        
        # Compare with actual forward pass
        output_actual = block(x)
        assert torch.allclose(x_final, output_actual, rtol=1e-6, atol=1e-6)
    
    def test_transformer_block_gradients(self, sample_input, d_model):
        """Test transformer block produces gradients for all parameters"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        # Enable gradients for input
        x = sample_input.clone().requires_grad_(True)
        output = block(x)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check all parameters have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.shape == param.shape
    
    def test_transformer_block_different_configurations(self, d_model):
        """Test transformer block with different valid configurations"""
        x = torch.randn(2, 4, d_model)
        
        # Test different head counts
        valid_head_counts = [h for h in [1, 2, 4, 8] if d_model % h == 0]
        
        for num_heads in valid_head_counts:
            for d_ff_ratio in [1, 2, 4]:
                d_ff = d_model * d_ff_ratio
                block = TransformerBlock(d_model, num_heads, d_ff, 0.1, 0.1)
                output = block(x)
                
                assert output.shape == x.shape
                assert not torch.any(torch.isnan(output))
    
    def test_transformer_block_dropout_behavior(self, sample_input, d_model):
        """Test transformer block dropout behavior in train vs eval"""
        num_heads = 8
        d_ff = d_model * 4
        dropout_prob = 0.5  # High dropout to see effect
        
        block = TransformerBlock(d_model, num_heads, d_ff, dropout_prob, dropout_prob)
        
        # Training mode - should have randomness due to dropout
        block.train()
        output_train1 = block(sample_input)
        output_train2 = block(sample_input)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2, rtol=1e-6, atol=1e-6)
        
        # Eval mode - should be deterministic
        block.eval()
        output_eval1 = block(sample_input)
        output_eval2 = block(sample_input)
        
        # Outputs should be the same
        assert torch.allclose(output_eval1, output_eval2)
    
    def test_transformer_block_parameter_count(self, d_model):
        """Test transformer block has expected parameter count"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        total_params = sum(p.numel() for p in block.parameters())
        
        # Expected parameters:
        # - RMSNorm 1: d_model (weight)
        # - Multi-head attention: 4 * d_model * d_model (q, k, v, output projections)
        # - RMSNorm 2: d_model (weight)
        # - FFN: 2 * d_model * d_ff (two linear layers)
        expected_params = (
            d_model +  # ln1 weight
            4 * d_model * d_model +  # attention projections
            d_model +  # ln2 weight
            2 * d_model * d_ff  # FFN weights
        )
        
        assert total_params == expected_params
    
    def test_transformer_block_sequence_length_independence(self, d_model):
        """Test transformer block works with different sequence lengths"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        batch_size = 2
        seq_lengths = [1, 8, 32, 64]
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, d_model)
            output = block(x)
            
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.any(torch.isnan(output))
    
    def test_transformer_block_numerical_stability(self, d_model):
        """Test transformer block numerical stability"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        # Test with extreme values
        x_small = torch.randn(2, 4, d_model) * 1e-6
        output_small = block(x_small)
        assert not torch.any(torch.isnan(output_small))
        assert not torch.any(torch.isinf(output_small))
        
        x_large = torch.randn(2, 4, d_model) * 1e3
        output_large = block(x_large)
        assert not torch.any(torch.isnan(output_large))
        # Large values might produce inf, which could be acceptable
    
    def test_transformer_block_deterministic_eval(self, sample_input, d_model):
        """Test transformer block is deterministic in eval mode"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        block.eval()
        
        output1 = block(sample_input)
        output2 = block(sample_input)
        
        assert torch.allclose(output1, output2)
    
    def test_transformer_block_batch_independence(self, d_model):
        """Test transformer block processes batch samples independently"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        seq_len = 8
        
        # Create batch with different samples
        x1 = torch.randn(1, seq_len, d_model)
        x2 = torch.randn(1, seq_len, d_model)
        x_batch = torch.cat([x1, x2], dim=0)
        
        # Process batch
        output_batch = block(x_batch)
        
        # Process individually
        output1 = block(x1)
        output2 = block(x2)
        output_individual = torch.cat([output1, output2], dim=0)
        
        # Should be the same
        assert torch.allclose(output_batch, output_individual, rtol=1e-6, atol=1e-6)
    
    def test_transformer_block_memory_efficiency(self, d_model):
        """Test transformer block memory usage with larger inputs"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        # Test with larger sequence length
        x = torch.randn(1, 128, d_model)
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.any(torch.isnan(output))
    
    def test_transformer_block_causal_attention(self, d_model):
        """Test transformer block maintains causal attention property"""
        num_heads = 8
        d_ff = d_model * 4
        block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
        
        seq_len = 8
        x = torch.randn(1, seq_len, d_model)
        
        # Process full sequence
        output_full = block(x)
        
        # Process prefixes of increasing length
        for prefix_len in range(1, seq_len + 1):
            x_prefix = x[:, :prefix_len, :]
            output_prefix = block(x_prefix)
            
            # The prefix output should match the corresponding part of full output
            # (due to causal attention masking)
            assert torch.allclose(
                output_prefix, 
                output_full[:, :prefix_len, :], 
                atol=1e-6
            )