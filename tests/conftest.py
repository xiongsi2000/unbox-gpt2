import pytest
import torch
import numpy as np
from .test_config import TestParameters, set_deterministic_seed


@pytest.fixture(autouse=True)
def setup_deterministic_testing():
    """Automatically set deterministic seed for all tests"""
    set_deterministic_seed(42)


@pytest.fixture
def device():
    """Test device - CPU for CI compatibility"""
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    """Standard batch size for tests following assignment 1 patterns"""
    return TestParameters.BATCH_SIZE


@pytest.fixture
def seq_len():
    """Standard sequence length for tests following assignment 1 patterns"""
    return TestParameters.SEQ_LEN


@pytest.fixture
def d_model():
    """Standard model dimension for tests following assignment 1 patterns"""
    return TestParameters.D_MODEL


@pytest.fixture
def d_ff():
    """Standard feed-forward dimension for tests"""
    return TestParameters.D_FF


@pytest.fixture
def num_heads():
    """Standard number of attention heads for tests"""
    return TestParameters.NUM_HEADS


@pytest.fixture
def vocab_size():
    """Standard vocabulary size for tests"""
    return TestParameters.VOCAB_SIZE


@pytest.fixture
def context_length():
    """Standard context length for tests"""
    return TestParameters.CONTEXT_LENGTH


@pytest.fixture
def sample_input(batch_size, seq_len, d_model, device):
    """Sample input tensor for testing with deterministic values"""
    torch.manual_seed(42)  # Ensure deterministic test data
    return torch.randn(batch_size, seq_len, d_model, device=device)


@pytest.fixture
def sample_tokens(batch_size, seq_len, vocab_size, device):
    """Sample token indices for testing with deterministic values"""
    torch.manual_seed(42)  # Ensure deterministic test data
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


@pytest.fixture
def deterministic_model_config(vocab_size, context_length, d_model, num_heads, d_ff):
    """Deterministic model configuration for testing"""
    return {
        'vocab_size': vocab_size,
        'context_length': context_length,
        'd_model': d_model,
        'num_layers': 2,
        'num_heads': num_heads,
        'd_ff': d_ff,
        'attn_dropout': 0.0,  # No dropout for deterministic testing
        'residual_dropout': 0.0
    }