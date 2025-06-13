"""
Test configuration and utilities following assignment 1 patterns
"""
import torch
from pathlib import Path

# Test tolerances following original assignment patterns
TOLERANCE_STRICT = 1e-6    # For simple mathematical operations
TOLERANCE_MEDIUM = 1e-4    # For complex multi-step operations  
TOLERANCE_LOOSE = 1e-3     # For stochastic or approximate operations

# Fixtures path
FIXTURES_PATH = Path(__file__).parent / "fixtures"

def set_deterministic_seed(seed=42):
    """Set deterministic seed for reproducible tests"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def assert_close_with_context(actual, expected, tolerance, context=""):
    """Enhanced assertion with better error messages"""
    if not torch.allclose(actual, expected, atol=tolerance):
        diff = torch.abs(actual - expected)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        error_msg = f"""
        {context}
        Tensors are not close enough!
        - Max difference: {max_diff:.2e}
        - Mean difference: {mean_diff:.2e}
        - Tolerance: {tolerance:.2e}
        - Actual shape: {actual.shape}
        - Expected shape: {expected.shape}
        """
        
        # Show a few sample values for debugging
        if actual.numel() <= 20:
            error_msg += f"\nActual values: {actual.flatten()[:10]}"
            error_msg += f"\nExpected values: {expected.flatten()[:10]}"
        
        raise AssertionError(error_msg)

class TestParameters:
    """Standard test parameters following assignment 1 patterns"""
    BATCH_SIZE = 2
    SEQ_LEN = 4
    D_MODEL = 8
    D_FF = 16
    NUM_HEADS = 2
    VOCAB_SIZE = 20
    CONTEXT_LENGTH = 8
    
    @classmethod
    def get_config_dict(cls):
        return {
            'batch_size': cls.BATCH_SIZE,
            'seq_len': cls.SEQ_LEN,
            'd_model': cls.D_MODEL,
            'd_ff': cls.D_FF,
            'num_heads': cls.NUM_HEADS,
            'vocab_size': cls.VOCAB_SIZE,
            'context_length': cls.CONTEXT_LENGTH
        }