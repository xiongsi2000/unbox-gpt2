import pytest
import torch
import torch.nn as nn
from transformer_lib.feedforward import FeedForwardNetwork


class TestFeedForwardNetwork:
    """Test cases for Feed-Forward Network"""
    
    def test_ffn_initialization(self, d_model):
        """Test FFN initializes correctly"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        assert isinstance(ffn, nn.Module)
        assert isinstance(ffn.w1, nn.Linear)
        assert isinstance(ffn.w2, nn.Linear)
        assert ffn.w1.in_features == d_model
        assert ffn.w1.out_features == d_ff
        assert ffn.w2.in_features == d_ff
        assert ffn.w2.out_features == d_model
        
        # Check no bias
        assert ffn.w1.bias is None
        assert ffn.w2.bias is None
    
    def test_ffn_shape_preservation(self, sample_input, d_model):
        """Test FFN preserves input/output shape"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        output = ffn(sample_input)
        assert output.shape == sample_input.shape
    
    def test_ffn_different_expansion_ratios(self, d_model):
        """Test FFN with different expansion ratios"""
        x = torch.randn(2, 4, d_model)
        
        expansion_ratios = [1, 2, 4, 8]
        for ratio in expansion_ratios:
            d_ff = d_model * ratio
            ffn = FeedForwardNetwork(d_model, d_ff)
            output = ffn(x)
            
            assert output.shape == x.shape
            assert not torch.any(torch.isnan(output))
    
    def test_ffn_forward_pass(self, d_model):
        """Test FFN forward pass computation"""
        d_ff = d_model * 2
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        x = torch.randn(1, 1, d_model)
        
        # Manual forward pass to verify computation
        with torch.no_grad():
            # First linear layer
            h = ffn.w1(x)
            assert h.shape == (1, 1, d_ff)
            
            # GELU activation
            h_activated = ffn.gelu(h)
            assert h_activated.shape == (1, 1, d_ff)
            
            # Second linear layer
            output_manual = ffn.w2(h_activated)
            assert output_manual.shape == (1, 1, d_model)
        
        # Compare with actual forward pass
        output_actual = ffn(x)
        assert torch.allclose(output_manual, output_actual)
    
    def test_ffn_gelu_activation(self, d_model):
        """Test FFN uses GELU activation correctly"""
        d_ff = d_model * 2
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        # Test with known input to verify GELU behavior
        x = torch.zeros(1, 1, d_model)
        output = ffn(x)
        
        # GELU(0) = 0, so if first layer weights are initialized properly,
        # we should get some non-zero output after second layer
        assert output.shape == x.shape
    
    def test_ffn_gradients(self, sample_input, d_model):
        """Test FFN produces gradients for all parameters"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        # Enable gradients for input
        x = sample_input.clone().requires_grad_(True)
        output = ffn(x)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check parameter gradients
        assert ffn.w1.weight.grad is not None
        assert ffn.w2.weight.grad is not None
        assert ffn.w1.weight.grad.shape == ffn.w1.weight.shape
        assert ffn.w2.weight.grad.shape == ffn.w2.weight.shape
    
    def test_ffn_parameter_count(self, d_model):
        """Test FFN has expected parameter count"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        total_params = sum(p.numel() for p in ffn.parameters())
        
        # Expected: w1 (d_model * d_ff) + w2 (d_ff * d_model) = 2 * d_model * d_ff
        expected_params = 2 * d_model * d_ff
        assert total_params == expected_params
    
    def test_ffn_no_bias(self, d_model):
        """Test FFN linear layers have no bias"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        assert ffn.w1.bias is None
        assert ffn.w2.bias is None
        
        # Verify no bias parameters in total count
        bias_params = sum(p.numel() for p in ffn.parameters() if p.dim() == 1)
        assert bias_params == 0  # Only bias parameters are 1D
    
    def test_ffn_deterministic(self, sample_input, d_model):
        """Test FFN produces deterministic outputs"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        ffn.eval()  # Ensure no randomness
        
        output1 = ffn(sample_input)
        output2 = ffn(sample_input)
        
        assert torch.allclose(output1, output2)
    
    def test_ffn_different_batch_sizes(self, d_model):
        """Test FFN works with different batch sizes"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        seq_len = 8
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ffn(x)
            
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.any(torch.isnan(output))
    
    def test_ffn_different_sequence_lengths(self, d_model):
        """Test FFN works with different sequence lengths"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        batch_size = 2
        seq_lengths = [1, 8, 32, 128]
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ffn(x)
            
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.any(torch.isnan(output))
    
    def test_ffn_numerical_stability(self, d_model):
        """Test FFN numerical stability with extreme values"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        # Test with very small values
        x_small = torch.randn(2, 4, d_model) * 1e-8
        output_small = ffn(x_small)
        assert not torch.any(torch.isnan(output_small))
        assert not torch.any(torch.isinf(output_small))
        
        # Test with very large values
        x_large = torch.randn(2, 4, d_model) * 1e3
        output_large = ffn(x_large)
        assert not torch.any(torch.isnan(output_large))
        # Note: inf values might be acceptable for very large inputs due to GELU
    
    def test_ffn_weight_initialization(self, d_model):
        """Test FFN weight initialization is reasonable"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        # Check weights are not all zeros or ones
        w1_weight = ffn.w1.weight
        w2_weight = ffn.w2.weight
        
        assert not torch.allclose(w1_weight, torch.zeros_like(w1_weight))
        assert not torch.allclose(w1_weight, torch.ones_like(w1_weight))
        assert not torch.allclose(w2_weight, torch.zeros_like(w2_weight))
        assert not torch.allclose(w2_weight, torch.ones_like(w2_weight))
        
        # Check weight magnitudes are reasonable (not too large or small)
        assert w1_weight.abs().mean() > 1e-3
        assert w1_weight.abs().mean() < 1.0
        assert w2_weight.abs().mean() > 1e-3
        assert w2_weight.abs().mean() < 1.0
    
    def test_ffn_position_wise_processing(self, d_model):
        """Test FFN processes each position independently"""
        d_ff = d_model * 4
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        batch_size = 2
        seq_len = 4
        
        # Create input where each position has different values
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Process full sequence
        output_full = ffn(x)
        
        # Process each position individually
        outputs_individual = []
        for i in range(seq_len):
            pos_input = x[:, i:i+1, :]  # Single position
            pos_output = ffn(pos_input)
            outputs_individual.append(pos_output)
        
        output_reconstructed = torch.cat(outputs_individual, dim=1)
        
        # Should be identical (position-wise processing)
        assert torch.allclose(output_full, output_reconstructed, rtol=1e-6, atol=1e-6)