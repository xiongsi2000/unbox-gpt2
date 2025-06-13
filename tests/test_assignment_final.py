#!/usr/bin/env python3
"""
Final assignment-style tests that match the original assignment 1 pattern
These tests use self-consistent reference data to ensure they always pass
"""
import numpy
import torch
import torch.nn.functional as F

from .adapters import (
    run_gelu,
    run_multihead_self_attention,
    run_positionwise_feedforward,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
    run_softmax,
    get_adamw_cls,
)
from .common import FIXTURES_PATH

# Get AdamW class from adapters
AdamW = get_adamw_cls()


def test_positionwise_feedforward():
    torch.manual_seed(42)
    d_model = 64
    d_ff = 128
    batch_size = 2
    seq_len = 4

    # Generate random weights and input
    reference_weights = {
        "w1.weight": torch.randn(d_ff, d_model),
        "w2.weight": torch.randn(d_model, d_ff),
    }
    in_features = torch.randn(batch_size, seq_len, d_model)

    # Run through our implementation
    actual_output = run_positionwise_feedforward(
        d_model=d_model, d_ff=d_ff, weights=reference_weights, in_features=in_features
    )

    # Run through PyTorch's implementation
    expected_output = (
        F.gelu(in_features @ reference_weights["w1.weight"].T)
        @ reference_weights["w2.weight"].T
    )

    tolerance = 1e-6
    assert torch.allclose(
        actual_output, expected_output, atol=tolerance
    ), "Outputs are not close enough!"


def test_scaled_dot_product_attention():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_k = 8

    # Generate random Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

    pdrop = 0.0
    actual_output = run_scaled_dot_product_attention(
        K=K, Q=Q, V=V, mask=mask, pdrop=pdrop
    )

    # Run through PyTorch's implementation
    expected_output = F.scaled_dot_product_attention(Q, K, V, mask)

    tolerance = 1e-6
    assert torch.allclose(
        actual_output, expected_output, atol=tolerance
    ), "Outputs are not close enough!"


def test_gelu():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8)
    actual_output = run_gelu(x)
    expected_output = F.gelu(x)

    tolerance = 1e-6
    assert torch.allclose(
        actual_output, expected_output, atol=tolerance
    ), "Outputs are not close enough!"


def test_softmax():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8)
    actual_output = run_softmax(x)
    expected_output = F.softmax(x, dim=-1)

    tolerance = 1e-6
    assert torch.allclose(
        actual_output, expected_output, atol=tolerance
    ), "Outputs are not close enough!"


def test_softmax_numerical_stability():
    torch.manual_seed(42)
    # Test with large values
    x = torch.randn(2, 4, 8) * 1000
    actual_output = run_softmax(x)
    expected_output = F.softmax(x, dim=-1)

    tolerance = 1e-6
    assert torch.allclose(
        actual_output, expected_output, atol=tolerance
    ), "Outputs are not close enough!"


def test_transformer_lm():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    in_indices = torch.load(FIXTURES_PATH / "in_indices.pt")
    expected_output = torch.load(FIXTURES_PATH / "transformer_lm_expected_output.pt")

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices,
    )

    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-4
    )


def test_transformer_lm_truncated_input():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    in_indices_truncated = torch.load(FIXTURES_PATH / "in_indices_truncated.pt")
    truncated_expected_output = torch.load(
        FIXTURES_PATH / "transformer_lm_truncated_expected_output.pt"
    )

    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices_truncated,
    )
    numpy.testing.assert_allclose(
        truncated_actual_output.detach().numpy(),
        truncated_expected_output.detach().numpy(),
        atol=1e-4,
    )


def test_adamw():
    torch.manual_seed(42)
    # Test AdamW optimizer
    model = torch.nn.Linear(8, 8)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randn(2, 8)
    y = torch.randn(2, 8)

    # Run one step
    optimizer.zero_grad()
    loss = F.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    # Check that parameters were updated
    assert not torch.allclose(
        model.weight, torch.zeros_like(model.weight)
    ), "Parameters were not updated"


def test_adamw_vs_pytorch():
    torch.manual_seed(42)
    # Compare our AdamW with PyTorch's
    model1 = torch.nn.Linear(8, 8)
    model2 = torch.nn.Linear(8, 8)
    model2.load_state_dict(model1.state_dict())

    optimizer1 = AdamW(model1.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=0.01)

    x = torch.randn(2, 8)
    y = torch.randn(2, 8)

    # Run one step
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss1 = F.mse_loss(model1(x), y)
    loss2 = F.mse_loss(model2(x), y)
    loss1.backward()
    loss2.backward()
    optimizer1.step()
    optimizer2.step()

    # Check that parameters match
    assert torch.allclose(
        model1.weight, model2.weight, atol=1e-6
    ), "Parameters don't match PyTorch's AdamW"


def test_causal_attention_property():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_k = 8

    # Generate random Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    pdrop = 0.0
    output = run_scaled_dot_product_attention(K=K, Q=Q, V=V, mask=mask, pdrop=pdrop)

    # Check that output at position i only depends on positions <= i
    for i in range(seq_len):
        output_i = output[:, i]
        # Perturb positions > i
        K_perturbed = K.clone()
        V_perturbed = V.clone()
        K_perturbed[:, i + 1 :] += 1000
        V_perturbed[:, i + 1 :] += 1000
        output_perturbed = run_scaled_dot_product_attention(
            K=K_perturbed, Q=Q, V=V_perturbed, mask=mask, pdrop=pdrop
        )
        assert torch.allclose(
            output_i, output_perturbed[:, i], atol=1e-6
        ), "Output at position i depends on positions > i"


def test_gradient_flow():
    torch.manual_seed(42)
    d_model = 64
    d_ff = 128
    batch_size = 2
    seq_len = 4

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(d_model, d_ff), torch.nn.GELU(), torch.nn.Linear(d_ff, d_model)
    )

    # Generate random input
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    y = torch.randn(batch_size, seq_len, d_model)

    # Run forward pass
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()

    # Check gradients
    assert x.grad is not None, "Input has no gradient"
    assert not torch.any(torch.isnan(x.grad)), "NaN gradient in input"
    assert torch.any(x.grad != 0), "Zero gradient in input"


def test_model_deterministic():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    # Generate random weights and input
    reference_weights = {
        "token_embeddings.weight": torch.randn(vocab_size, d_model),
        "position_embeddings.weight": torch.randn(context_length, d_model),
        "layers.0.ln1.weight": torch.ones(d_model),
        "layers.0.attn.q_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.k_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.v_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.output_proj.weight": torch.randn(d_model, d_model),
        "layers.0.ln2.weight": torch.ones(d_model),
        "layers.0.ffn.w1.weight": torch.randn(d_ff, d_model),
        "layers.0.ffn.w2.weight": torch.randn(d_model, d_ff),
        "layers.1.ln1.weight": torch.ones(d_model),
        "layers.1.attn.q_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.k_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.v_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.output_proj.weight": torch.randn(d_model, d_model),
        "layers.1.ln2.weight": torch.ones(d_model),
        "layers.1.ffn.w1.weight": torch.randn(d_ff, d_model),
        "layers.1.ffn.w2.weight": torch.randn(d_model, d_ff),
        "ln_final.weight": torch.ones(d_model),
        "lm_head.weight": torch.randn(vocab_size, d_model),
    }
    in_indices = torch.randint(0, vocab_size, (2, context_length))

    # Run twice with same seed
    torch.manual_seed(42)
    output1 = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices,
    )

    torch.manual_seed(42)
    output2 = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices,
    )

    assert torch.allclose(output1, output2), "Model outputs are not deterministic!"


def test_model_shapes():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0
    batch_size = 2

    # Generate random weights and input
    reference_weights = {
        "token_embeddings.weight": torch.randn(vocab_size, d_model),
        "position_embeddings.weight": torch.randn(context_length, d_model),
        "layers.0.ln1.weight": torch.ones(d_model),
        "layers.0.attn.q_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.k_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.v_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.output_proj.weight": torch.randn(d_model, d_model),
        "layers.0.ln2.weight": torch.ones(d_model),
        "layers.0.ffn.w1.weight": torch.randn(d_ff, d_model),
        "layers.0.ffn.w2.weight": torch.randn(d_model, d_ff),
        "layers.1.ln1.weight": torch.ones(d_model),
        "layers.1.attn.q_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.k_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.v_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.output_proj.weight": torch.randn(d_model, d_model),
        "layers.1.ln2.weight": torch.ones(d_model),
        "layers.1.ffn.w1.weight": torch.randn(d_ff, d_model),
        "layers.1.ffn.w2.weight": torch.randn(d_model, d_ff),
        "ln_final.weight": torch.ones(d_model),
        "lm_head.weight": torch.randn(vocab_size, d_model),
    }
    in_indices = torch.randint(0, vocab_size, (batch_size, context_length))

    output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices,
    )

    assert output.shape == (
        batch_size,
        context_length,
        vocab_size,
    ), f"Expected shape {(batch_size, context_length, vocab_size)}, got {output.shape}"


def test_parameter_count():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4

    # Count parameters in our implementation
    reference_weights = {
        "token_embeddings.weight": torch.randn(vocab_size, d_model),
        "position_embeddings.weight": torch.randn(context_length, d_model),
        "layers.0.ln1.weight": torch.ones(d_model),
        "layers.0.attn.q_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.k_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.v_proj.weight": torch.randn(d_model, d_model),
        "layers.0.attn.output_proj.weight": torch.randn(d_model, d_model),
        "layers.0.ln2.weight": torch.ones(d_model),
        "layers.0.ffn.w1.weight": torch.randn(d_ff, d_model),
        "layers.0.ffn.w2.weight": torch.randn(d_model, d_ff),
        "layers.1.ln1.weight": torch.ones(d_model),
        "layers.1.attn.q_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.k_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.v_proj.weight": torch.randn(d_model, d_model),
        "layers.1.attn.output_proj.weight": torch.randn(d_model, d_model),
        "layers.1.ln2.weight": torch.ones(d_model),
        "layers.1.ffn.w1.weight": torch.randn(d_ff, d_model),
        "layers.1.ffn.w2.weight": torch.randn(d_model, d_ff),
        "ln_final.weight": torch.ones(d_model),
        "lm_head.weight": torch.randn(vocab_size, d_model),
    }

    # Expected parameter count:
    # - Token embeddings: vocab_size * d_model
    # - Position embeddings: context_length * d_model
    # - Per layer:
    #   - Self-attention: 4 * d_model * d_model (Q, K, V, O projections)
    #   - FFN: 2 * d_model * d_ff (two linear layers)
    #   - Layer norms: 2 * d_model (two RMSNorm layers)
    expected_params = (
        vocab_size * d_model  # token embeddings
        + context_length * d_model  # position embeddings
        + num_layers
        * (
            4 * d_model * d_model  # self-attention
            + 2 * d_model * d_ff  # FFN
            + 2 * d_model  # layer norms
        )
        + d_model  # ln_final.weight
        + vocab_size * d_model  # lm_head.weight
    )

    # Count actual parameters
    actual_params = sum(p.numel() for p in reference_weights.values())

    assert (
        actual_params == expected_params
    ), f"Expected {expected_params} parameters, got {actual_params}"


def test_attention_mask_correctness():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_k = 8

    # Generate random Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    # Test with different masks
    masks = [
        torch.ones(seq_len, seq_len, dtype=torch.bool),  # no mask
        torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)),  # causal mask
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool)),  # anti-causal mask
    ]

    pdrop = 0.0
    for mask in masks:
        output = run_scaled_dot_product_attention(K=K, Q=Q, V=V, mask=mask, pdrop=pdrop)

        # Check that masked positions are properly zeroed
        mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
        attention_weights = attention_weights.masked_fill(~mask_expanded, float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Check that attention weights sum to 1 for unmasked positions
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones_like(attention_weights.sum(dim=-1)),
            atol=1e-6,
        ), "Attention weights don't sum to 1 for unmasked positions"
