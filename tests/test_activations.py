import pytest
import torch
import torch.nn as nn
import math
from transformer_lib.activations import GELU, Softmax


class TestGELU:
    """Test cases for GELU activation function"""
    
    def test_gelu_initialization(self):
        """Test GELU can be initialized properly"""
        gelu = GELU()
        assert isinstance(gelu, nn.Module)
    
    def test_gelu_forward_shape(self, sample_input):
        """Test GELU preserves input shape"""
        gelu = GELU()
        output = gelu(sample_input)
        assert output.shape == sample_input.shape
    
    def test_gelu_forward_values(self):
        """Test GELU produces expected values for known inputs"""
        gelu = GELU()
        
        # Test zero input
        x = torch.tensor([0.0])
        output = gelu(x)
        expected = 0.0
        assert torch.allclose(output, torch.tensor([expected]), rtol=1e-6, atol=1e-6)
        
        # Test positive input
        x = torch.tensor([1.0])
        output = gelu(x)
        expected = 0.5 * 1.0 * (1 + torch.erf(torch.tensor(1.0) / math.sqrt(2.0)))
        assert torch.allclose(output, expected, rtol=1e-6, atol=1e-6)
        
        # Test negative input
        x = torch.tensor([-1.0])
        output = gelu(x)
        expected = 0.5 * (-1.0) * (1 + torch.erf(torch.tensor(-1.0) / math.sqrt(2.0)))
        assert torch.allclose(output, expected, rtol=1e-6, atol=1e-6)
    
    def test_gelu_differentiable(self):
        """Test GELU is differentiable"""
        gelu = GELU()
        x = torch.tensor([1.0], requires_grad=True)
        output = gelu(x)
        output.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_gelu_monotonic_positive(self):
        """Test GELU is approximately monotonic for positive values"""
        gelu = GELU()
        x = torch.linspace(0.1, 5.0, 100)
        output = gelu(x)
        
        # Check that output is generally increasing
        diff = output[1:] - output[:-1]
        assert torch.all(diff >= -1e-6)  # Allow tiny numerical errors


class TestSoftmax:
    """Test cases for numerically stable Softmax"""
    
    def test_softmax_initialization(self):
        """Test Softmax can be initialized with different dimensions"""
        softmax_last = Softmax(dim=-1)
        softmax_first = Softmax(dim=0)
        assert isinstance(softmax_last, nn.Module)
        assert isinstance(softmax_first, nn.Module)
    
    def test_softmax_shape_preservation(self, sample_input):
        """Test Softmax preserves input shape"""
        softmax = Softmax(dim=-1)
        output = softmax(sample_input)
        assert output.shape == sample_input.shape
    
    def test_softmax_sum_to_one(self, sample_input):
        """Test Softmax outputs sum to 1 along specified dimension"""
        softmax = Softmax(dim=-1)
        output = softmax(sample_input)
        
        # Sum along last dimension should be 1
        sums = output.sum(dim=-1)
        expected = torch.ones_like(sums)
        assert torch.allclose(sums, expected, rtol=1e-6, atol=1e-6)
    
    def test_softmax_non_negative(self, sample_input):
        """Test Softmax outputs are non-negative"""
        softmax = Softmax(dim=-1)
        output = softmax(sample_input)
        assert torch.all(output >= 0)
    
    def test_softmax_numerical_stability(self):
        """Test Softmax handles large values without overflow"""
        softmax = Softmax(dim=-1)
        
        # Test with large values that would cause overflow in naive implementation
        x = torch.tensor([[1000.0, 1001.0, 1002.0]])
        output = softmax(x)
        
        # Should not contain NaN or Inf
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
        
        # Should still sum to 1
        assert torch.allclose(output.sum(dim=-1), torch.tensor([1.0]), rtol=1e-6, atol=1e-6)
    
    def test_softmax_different_dimensions(self):
        """Test Softmax works correctly on different dimensions"""
        x = torch.randn(2, 3, 4)
        
        # Test dim=0
        softmax_0 = Softmax(dim=0)
        output_0 = softmax_0(x)
        assert torch.allclose(output_0.sum(dim=0), torch.ones(3, 4), rtol=1e-6, atol=1e-6)
        
        # Test dim=1
        softmax_1 = Softmax(dim=1)
        output_1 = softmax_1(x)
        assert torch.allclose(output_1.sum(dim=1), torch.ones(2, 4), rtol=1e-6, atol=1e-6)
        
        # Test dim=2
        softmax_2 = Softmax(dim=2)
        output_2 = softmax_2(x)
        assert torch.allclose(output_2.sum(dim=2), torch.ones(2, 3), rtol=1e-6, atol=1e-6)
    
    def test_softmax_gradients(self):
        """Test Softmax produces correct gradients"""
        softmax = Softmax(dim=-1)
        x = torch.randn(2, 3, requires_grad=True)
        output = softmax(x)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.any(torch.isnan(x.grad))
    
    def test_softmax_extreme_values(self):
        """Test Softmax with extreme input values"""
        softmax = Softmax(dim=-1)
        
        # Test with very negative values
        x_neg = torch.tensor([[-1000.0, -1001.0, -1002.0]])
        output_neg = softmax(x_neg)
        assert not torch.any(torch.isnan(output_neg))
        assert torch.allclose(output_neg.sum(dim=-1), torch.tensor([1.0]), rtol=1e-6, atol=1e-6)
        
        # Test with mixed extreme values
        x_mixed = torch.tensor([[-1000.0, 0.0, 1000.0]])
        output_mixed = softmax(x_mixed)
        assert not torch.any(torch.isnan(output_mixed))
        assert torch.allclose(output_mixed.sum(dim=-1), torch.tensor([1.0]), rtol=1e-6, atol=1e-6)