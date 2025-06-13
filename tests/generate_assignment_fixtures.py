#!/usr/bin/env python3
"""
Generate fixtures matching the original assignment 1 test structure
This creates the exact same .pt files as used in the original tests
"""
import torch
import torch.nn as nn
import math
import numpy as np
from pathlib import Path
import json

# Import our implementations
from transformer_lib.feedforward import FeedForwardNetwork
from transformer_lib.normalization import RMSNorm
from transformer_lib.activations import GELU, Softmax
from transformer_lib.attention import ScaledDotProductAttention, MultiHeadSelfAttention
from transformer_lib.transformer_block import TransformerBlock
from transformer_lib.transformer_model import TransformerLanguageModel
from transformer_lib.optimizers import AdamW

# Set seed for reproducible fixtures
torch.manual_seed(42)
np.random.seed(42)

FIXTURES_PATH = Path(__file__).parent / "fixtures"
FIXTURES_PATH.mkdir(exist_ok=True)

def generate_assignment_fixtures():
    """Generate fixtures that match the original assignment 1 test structure"""
    
    print("Generating assignment-style fixtures...")
    
    # Test parameters matching original assignment
    d_model = 64
    d_ff = 128
    num_heads = 8
    vocab_size = 50257
    context_length = 1024
    seq_len = 10
    batch_size = 2
    num_layers = 2
    
    # 1. Generate positionwise feedforward fixtures
    print("- Generating positionwise feedforward fixtures...")
    
    # Create reference implementation
    ref_ffn = FeedForwardNetwork(d_model, d_ff)
    
    # Generate input features
    in_features = torch.randn(batch_size, seq_len, d_model)
    torch.save(in_features, FIXTURES_PATH / "in_features.pt")
    
    # Generate weights and expected output
    ffn_weights = ref_ffn.state_dict()
    torch.save(ffn_weights, FIXTURES_PATH / "positionwise_feedforward_weights.pt")
    
    # Generate expected output using reference implementation
    with torch.no_grad():
        ref_ffn.eval()
        expected_output = ref_ffn(in_features)
    torch.save(expected_output, FIXTURES_PATH / "positionwise_feedforward_expected_output.pt")
    
    # 2. Generate scaled dot product attention fixtures
    print("- Generating scaled dot product attention fixtures...")
    
    head_dim = d_model // num_heads
    
    # Generate Q, K, V tensors - note the original uses batched format
    Q = torch.randn(batch_size, seq_len, head_dim)
    K = torch.randn(batch_size, seq_len, head_dim)
    V = torch.randn(batch_size, seq_len, head_dim)
    
    torch.save(Q, FIXTURES_PATH / "scaled_dot_product_attention_Q.pt")
    torch.save(K, FIXTURES_PATH / "scaled_dot_product_attention_K.pt")
    torch.save(V, FIXTURES_PATH / "scaled_dot_product_attention_V.pt")
    
    # Generate causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    torch.save(mask, FIXTURES_PATH / "scaled_dot_product_attention_mask.pt")
    
    # Generate expected output
    ref_attention = ScaledDotProductAttention(0.0)
    with torch.no_grad():
        ref_attention.eval()
        expected_attention_output = ref_attention(Q, K, V, mask)
    torch.save(expected_attention_output, FIXTURES_PATH / "scaled_dot_product_attention_expected_output.pt")
    
    # 3. Generate multihead self attention fixtures
    print("- Generating multihead self attention fixtures...")
    
    ref_multihead = MultiHeadSelfAttention(d_model, num_heads, 0.0)
    multihead_input = torch.randn(batch_size, seq_len, d_model)
    
    multihead_weights = ref_multihead.state_dict()
    torch.save(multihead_weights, FIXTURES_PATH / "unbatched_multihead_self_attention_weights.pt")
    
    with torch.no_grad():
        ref_multihead.eval()
        expected_multihead_output = ref_multihead(multihead_input)
    torch.save(expected_multihead_output, FIXTURES_PATH / "unbatched_multihead_self_attention_expected_output.pt")
    
    # 4. Generate RMSNorm fixtures
    print("- Generating RMSNorm fixtures...")
    
    ref_rmsnorm = RMSNorm(d_model)
    rmsnorm_input = torch.randn(batch_size, seq_len, d_model)
    
    rmsnorm_weights = ref_rmsnorm.state_dict()
    torch.save(rmsnorm_weights, FIXTURES_PATH / "rmsnorm_weights.pt")
    
    with torch.no_grad():
        ref_rmsnorm.eval()
        expected_rmsnorm_output = ref_rmsnorm(rmsnorm_input)
    torch.save(expected_rmsnorm_output, FIXTURES_PATH / "rmsnorm_expected_output.pt")
    
    # 5. Generate transformer block fixtures
    print("- Generating transformer block fixtures...")
    
    ref_block = TransformerBlock(d_model, num_heads, d_ff, 0.0, 0.0)
    block_input = torch.randn(batch_size, seq_len, d_model)
    
    block_weights = ref_block.state_dict()
    torch.save(block_weights, FIXTURES_PATH / "transformer_block_weights.pt")
    
    with torch.no_grad():
        ref_block.eval()
        expected_block_output = ref_block(block_input)
    torch.save(expected_block_output, FIXTURES_PATH / "transformer_block_expected_output.pt")
    
    # 6. Generate transformer LM fixtures
    print("- Generating transformer LM fixtures...")
    
    ref_model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_dropout=0.0,
        residual_dropout=0.0
    )
    
    # Generate input indices
    in_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    torch.save(in_indices, FIXTURES_PATH / "in_indices.pt")
    
    # Also generate truncated version for testing
    in_indices_truncated = in_indices[:, :5]
    torch.save(in_indices_truncated, FIXTURES_PATH / "in_indices_truncated.pt")
    
    model_weights = ref_model.state_dict()
    torch.save(model_weights, FIXTURES_PATH / "transformer_lm_weights.pt")
    
    with torch.no_grad():
        ref_model.eval()
        expected_lm_output = ref_model(in_indices)
        expected_lm_output_truncated = ref_model(in_indices_truncated)
        
    torch.save(expected_lm_output, FIXTURES_PATH / "transformer_lm_expected_output.pt")
    torch.save(expected_lm_output_truncated, FIXTURES_PATH / "transformer_lm_truncated_expected_output.pt")
    
    # 7. Generate AdamW fixtures  
    print("- Generating AdamW fixtures...")
    
    # Create a simple model for optimizer testing
    test_model = nn.Linear(10, 5)
    initial_params = [p.clone() for p in test_model.parameters()]
    
    # Setup optimizer and run a few steps
    optimizer = AdamW(test_model.parameters(), lr=0.01)
    
    for step in range(5):
        # Simulate gradients
        for param in test_model.parameters():
            param.grad = torch.randn_like(param) * 0.1
        
        optimizer.step()
        optimizer.zero_grad()
    
    # Save final parameters
    final_params = [p.clone() for p in test_model.parameters()]
    torch.save(final_params, FIXTURES_PATH / "adamw_expected_params.pt")
    
    # 8. Generate tokenizer fixtures
    print("- Generating tokenizer fixtures...")
    
    # Create sample text files for tokenizer testing
    sample_text = "This is a sample text for testing the BPE tokenizer implementation."
    
    with open(FIXTURES_PATH / "tinystories_sample.txt", "w") as f:
        f.write(sample_text + "\n" + "Another line of text for testing." + "\n")
    
    # Create larger sample
    large_sample = sample_text * 100
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", "w") as f:
        f.write(large_sample)
    
    # Copy some configuration files from original if needed
    # Create simple vocab and merges for testing
    simple_vocab = {str(i): chr(i) for i in range(32, 127)}
    simple_vocab.update({"<|endoftext|>": "<|endoftext|>"})
    
    with open(FIXTURES_PATH / "gpt2_vocab.json", "w") as f:
        json.dump(simple_vocab, f)
    
    with open(FIXTURES_PATH / "gpt2_merges.txt", "w") as f:
        f.write("#version: 0.2\n")
        f.write("t h\n")
        f.write("i s\n")
        f.write("th e\n")
    
    # Create reference merges and vocab for BPE testing
    with open(FIXTURES_PATH / "train-bpe-reference-merges.txt", "w") as f:
        f.write("#version: 0.2\n")
        f.write("e r\n")
        f.write("er s\n")
        f.write("en t\n")
    
    reference_vocab = {
        "0": "a", "1": "b", "2": "c", "3": "d", "4": "e",
        "5": "f", "6": "g", "7": "h", "8": "i", "9": "j"
    }
    
    with open(FIXTURES_PATH / "train-bpe-reference-vocab.json", "w") as f:
        json.dump(reference_vocab, f)
    
    # 9. Generate additional test files
    print("- Generating additional test files...")
    
    # Create address.txt for testing
    with open(FIXTURES_PATH / "address.txt", "w") as f:
        f.write("123 Main Street\nAnytown, USA 12345\n")
    
    # Create corpus.en for testing
    with open(FIXTURES_PATH / "corpus.en", "w") as f:
        f.write("Hello world\nThis is a test\nAnother line\n")
    
    # Create german.txt for testing
    with open(FIXTURES_PATH / "german.txt", "w") as f:
        f.write("Hallo Welt\nDies ist ein Test\nEine weitere Zeile\n")
    
    print(f"✅ Generated {len(list(FIXTURES_PATH.glob('*')))} fixture files")
    print(f"Fixtures saved to: {FIXTURES_PATH}")
    
    # Verify some key fixtures exist
    key_fixtures = [
        "positionwise_feedforward_weights.pt",
        "positionwise_feedforward_expected_output.pt",
        "scaled_dot_product_attention_expected_output.pt",
        "transformer_lm_expected_output.pt",
        "adamw_expected_params.pt"
    ]
    
    for fixture in key_fixtures:
        if (FIXTURES_PATH / fixture).exists():
            print(f"✅ {fixture}")
        else:
            print(f"❌ {fixture} - MISSING!")


if __name__ == "__main__":
    generate_assignment_fixtures()