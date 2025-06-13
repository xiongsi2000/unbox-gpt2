import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy
from transformer_lib.activations import GELU, Softmax
from transformer_lib.normalization import RMSNorm
from transformer_lib.feedforward import FeedForwardNetwork
from transformer_lib.attention import MultiHeadSelfAttention
from transformer_lib.transformer_block import TransformerBlock
from transformer_lib.transformer_model import TransformerLanguageModel
from transformer_lib.optimizers import AdamW
from transformer_lib.utils import save_checkpoint, load_checkpoint

# Path to test fixtures
FIXTURES_PATH = Path(__file__).parent / "fixtures"


def test_pytorch_compatibility():
    """Test compatibility with PyTorch's implementations where applicable"""
    # Test GELU compatibility
    input_tensor = torch.randn(2, 4, 8)
    
    our_gelu = GELU()
    pytorch_gelu = nn.GELU()
    
    our_output = our_gelu(input_tensor)
    pytorch_output = pytorch_gelu(input_tensor)
    
    # Should be very close to PyTorch's implementation
    assert torch.allclose(our_output, pytorch_output, rtol=1e-6, atol=1e-6), \
        "Our GELU implementation differs from PyTorch's"
    
    # Test Softmax compatibility
    our_softmax = Softmax(dim=-1)
    pytorch_softmax = nn.Softmax(dim=-1)
    
    our_softmax_output = our_softmax(input_tensor)
    pytorch_softmax_output = pytorch_softmax(input_tensor)
    
    assert torch.allclose(our_softmax_output, pytorch_softmax_output, rtol=1e-6, atol=1e-6), \
        "Our Softmax implementation differs from PyTorch's"


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

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_dropout=attn_pdrop,
        residual_dropout=residual_pdrop
    )
    model.load_state_dict(reference_weights)
    actual_output = model(in_indices)

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

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_dropout=attn_pdrop,
        residual_dropout=residual_pdrop
    )
    model.load_state_dict(reference_weights)
    truncated_actual_output = model(in_indices_truncated)

    numpy.testing.assert_allclose(
        truncated_actual_output.detach().numpy(),
        truncated_expected_output.detach().numpy(),
        atol=1e-4,
    )


def test_transformer_block():
    torch.manual_seed(42)
    reference_weights = torch.load(FIXTURES_PATH / "transformer_block_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(FIXTURES_PATH / "transformer_block_expected_output.pt")
    d_model = 64
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    transformer_block = TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
    transformer_block.load_state_dict(reference_weights)
    actual_output = transformer_block(in_features)

    tolerance = 1e-6
    assert torch.allclose(actual_output, expected_output, atol=tolerance), "Outputs are not close enough!"


class _TestNet(nn.Module):
    def __init__(self, d_input: int = 100, d_output: int = 10):
        super(_TestNet, self).__init__()
        self.fc1 = nn.Linear(d_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, d_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def are_optimizers_equal(
    optimizer1_state_dict, optimizer2_state_dict, atol=1e-8, rtol=1e-5
):
    # Check if the keys of the main dictionaries are equal (e.g., 'state', 'param_groups')
    if set(optimizer1_state_dict.keys()) != set(optimizer2_state_dict.keys()):
        return False

    # Check parameter groups are identical
    if optimizer1_state_dict["param_groups"] != optimizer2_state_dict["param_groups"]:
        return False

    # Check states
    state1 = optimizer1_state_dict["state"]
    state2 = optimizer2_state_dict["state"]
    if set(state1.keys()) != set(state2.keys()):
        return False

    for key in state1:
        # Assuming state contents are also dictionaries
        if set(state1[key].keys()) != set(state2[key].keys()):
            return False

        for sub_key in state1[key]:
            item1 = state1[key][sub_key]
            item2 = state2[key][sub_key]

            # If both items are tensors, use torch.allclose
            if torch.is_tensor(item1) and torch.is_tensor(item2):
                if not torch.allclose(item1, item2, atol=atol, rtol=rtol):
                    return False
            # For non-tensor items, check for direct equality
            elif item1 != item2:
                return False
    return True


def test_checkpointing(tmp_path):
    torch.manual_seed(42)
    d_input = 100
    d_output = 10
    num_iters = 10

    model = _TestNet(d_input=d_input, d_output=d_output)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    it = 0
    for _ in range(num_iters):
        optimizer.zero_grad()
        x = torch.rand(d_input)
        y = torch.rand(d_output)
        y_hat = model(x)
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        optimizer.step()
        it += 1

    serialization_path = tmp_path / "checkpoint.pt"
    # Save the model
    save_checkpoint(
        model,
        optimizer,
        iteration=it,
        out=serialization_path,
    )

    # Load the model back again
    new_model = _TestNet(d_input=d_input, d_output=d_output)
    new_optimizer = AdamW(
        new_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loaded_iterations = load_checkpoint(
        checkpoint_path=serialization_path, model=new_model, optimizer=new_optimizer
    )
    assert it == loaded_iterations

    # Compare the loaded model state with the original model state
    original_model_state = model.state_dict()
    original_optimizer_state = optimizer.state_dict()
    new_model_state = new_model.state_dict()
    new_optimizer_state = new_optimizer.state_dict()

    # Check that state dict keys match
    assert set(original_model_state.keys()) == set(new_model_state.keys())
    assert set(original_optimizer_state.keys()) == set(new_optimizer_state.keys())

    # compare the model state dicts
    for key in original_model_state.keys():
        numpy.testing.assert_allclose(
            original_model_state[key].detach().numpy(),
            new_model_state[key].detach().numpy(),
        )
    # compare the optimizer state dicts
    assert are_optimizers_equal(original_optimizer_state, new_optimizer_state)
