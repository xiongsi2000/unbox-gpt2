import pytest
import torch
import torch.nn as nn
from transformer_lib.transformer_model import TransformerLanguageModel


class TestTransformerLanguageModel:
    """Test cases for complete Transformer Language Model"""
    
    @pytest.fixture
    def model_config(self, vocab_size, d_model):
        """Standard model configuration for testing"""
        return {
            'vocab_size': vocab_size,
            'context_length': 32,
            'd_model': d_model,
            'num_layers': 2,
            'num_heads': 8,
            'd_ff': d_model * 4,
            'attn_dropout': 0.0,
            'residual_dropout': 0.0
        }
    
    def test_model_initialization(self, model_config):
        """Test transformer model initializes correctly"""
        model = TransformerLanguageModel(**model_config)
        
        assert isinstance(model, nn.Module)
        assert model.vocab_size == model_config['vocab_size']
        assert model.context_length == model_config['context_length']
        assert model.d_model == model_config['d_model']
        assert len(model.layers) == model_config['num_layers']
        
        # Check embedding layers
        assert model.token_embeddings.num_embeddings == model_config['vocab_size']
        assert model.token_embeddings.embedding_dim == model_config['d_model']
        assert model.position_embeddings.num_embeddings == model_config['context_length']
        assert model.position_embeddings.embedding_dim == model_config['d_model']
        
        # Check output layer
        assert model.lm_head.in_features == model_config['d_model']
        assert model.lm_head.out_features == model_config['vocab_size']
        assert model.lm_head.bias is None
    
    def test_model_forward_shape(self, sample_tokens, model_config):
        """Test model forward pass produces correct output shape"""
        model = TransformerLanguageModel(**model_config)
        
        logits = model(sample_tokens)
        batch_size, seq_len = sample_tokens.shape
        
        expected_shape = (batch_size, seq_len, model_config['vocab_size'])
        assert logits.shape == expected_shape
    
    def test_model_forward_computation(self, model_config):
        """Test model forward pass step by step"""
        model = TransformerLanguageModel(**model_config)
        model.eval()  # Set to eval mode for deterministic behavior
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        
        # Manual forward pass
        with torch.no_grad():
            # Embeddings
            token_emb = model.token_embeddings(input_ids)
            pos_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            pos_emb = model.position_embeddings(pos_indices)
            x = token_emb + pos_emb
            x = model.dropout(x)
            
            # Transformer layers
            for layer in model.layers:
                x = layer(x)
            
            # Final norm and projection
            x = model.ln_final(x)
            logits_manual = model.lm_head(x)
        
        # Compare with actual forward pass
        logits_actual = model(input_ids)
        assert torch.allclose(logits_manual, logits_actual, rtol=1e-6, atol=1e-6)
    
    def test_model_different_sequence_lengths(self, model_config):
        """Test model works with different sequence lengths"""
        model = TransformerLanguageModel(**model_config)
        
        batch_size = 2
        max_len = model_config['context_length']
        
        # Test various sequence lengths up to context_length
        for seq_len in [1, 4, 8, 16, max_len]:
            input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
            logits = model(input_ids)
            
            assert logits.shape == (batch_size, seq_len, model_config['vocab_size'])
            assert not torch.any(torch.isnan(logits))
    
    def test_model_gradients(self, sample_tokens, model_config):
        """Test model produces gradients for all parameters"""
        model = TransformerLanguageModel(**model_config)
        
        logits = model(sample_tokens)
        loss = logits.sum()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.shape == param.shape
            assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"
    
    def test_model_parameter_count(self, model_config):
        """Test model has reasonable parameter count"""
        model = TransformerLanguageModel(**model_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Rough estimate of expected parameters
        vocab_size = model_config['vocab_size']
        context_length = model_config['context_length']
        d_model = model_config['d_model']
        num_layers = model_config['num_layers']
        d_ff = model_config['d_ff']
        
        # Token embeddings + position embeddings
        embedding_params = vocab_size * d_model + context_length * d_model
        
        # Each transformer layer: 2 RMSNorm + attention + FFN
        per_layer_params = (
            2 * d_model +  # 2 RMSNorm weights
            4 * d_model * d_model +  # attention projections
            2 * d_model * d_ff  # FFN weights
        )
        transformer_params = num_layers * per_layer_params
        
        # Final norm + language model head
        output_params = d_model + d_model * vocab_size
        
        expected_params = embedding_params + transformer_params + output_params
        
        # Allow some tolerance for implementation differences
        assert abs(total_params - expected_params) / expected_params < 0.1
    
    def test_model_generate_method(self, model_config):
        """Test model text generation method"""
        model = TransformerLanguageModel(**model_config)
        model.eval()
        
        # Test generation
        prompt = torch.randint(0, model_config['vocab_size'], (1, 5))
        max_length = 20
        
        generated = model.generate(prompt, max_length=max_length, temperature=1.0)
        
        assert generated.shape[0] == 1  # batch size
        assert generated.shape[1] <= max_length
        assert generated.shape[1] >= prompt.shape[1]
        
        # Check that generated tokens are valid
        assert torch.all(generated >= 0)
        assert torch.all(generated < model_config['vocab_size'])
        
        # Check that prompt is preserved
        assert torch.allclose(generated[:, :prompt.shape[1]], prompt)
    
    def test_model_generate_different_temperatures(self, model_config):
        """Test model generation with different temperatures"""
        model = TransformerLanguageModel(**model_config)
        model.eval()
        
        prompt = torch.randint(0, model_config['vocab_size'], (1, 3))
        max_length = 10
        
        # Test different temperatures
        temperatures = [0.1, 1.0, 2.0]
        generations = []
        
        for temp in temperatures:
            torch.manual_seed(42)  # For reproducibility
            generated = model.generate(prompt, max_length=max_length, temperature=temp)
            generations.append(generated)
            
            assert generated.shape[1] <= max_length
            assert torch.allclose(generated[:, :prompt.shape[1]], prompt)
        
        # Different temperatures should potentially produce different outputs
        # (though with small sequences, they might be the same by chance)
    
    def test_model_generate_top_k(self, model_config):
        """Test model generation with top-k sampling"""
        model = TransformerLanguageModel(**model_config)
        model.eval()
        
        prompt = torch.randint(0, model_config['vocab_size'], (1, 3))
        max_length = 10
        
        # Test with different top_k values
        for top_k in [1, 5, 10]:
            generated = model.generate(
                prompt, 
                max_length=max_length, 
                temperature=1.0, 
                top_k=top_k
            )
            
            assert generated.shape[1] <= max_length
            assert torch.allclose(generated[:, :prompt.shape[1]], prompt)
            assert torch.all(generated >= 0)
            assert torch.all(generated < model_config['vocab_size'])
    
    def test_model_context_length_limit(self, model_config):
        """Test model respects context length limits"""
        model = TransformerLanguageModel(**model_config)
        
        context_length = model_config['context_length']
        
        # Test at exactly context length
        input_ids = torch.randint(0, model_config['vocab_size'], (1, context_length))
        logits = model(input_ids)
        assert logits.shape == (1, context_length, model_config['vocab_size'])
        
        # Test generation doesn't exceed context length
        prompt = torch.randint(0, model_config['vocab_size'], (1, context_length - 5))
        generated = model.generate(prompt, max_length=context_length + 10)
        assert generated.shape[1] <= context_length
    
    def test_model_deterministic_eval(self, sample_tokens, model_config):
        """Test model is deterministic in eval mode"""
        model = TransformerLanguageModel(**model_config)
        model.eval()
        
        logits1 = model(sample_tokens)
        logits2 = model(sample_tokens)
        
        assert torch.allclose(logits1, logits2)
    
    def test_model_different_configurations(self, vocab_size, d_model):
        """Test model with different valid configurations"""
        base_config = {
            'vocab_size': vocab_size,
            'context_length': 16,
            'd_model': d_model,
            'num_layers': 1,
            'num_heads': 4,
            'd_ff': d_model * 2,
            'attn_dropout': 0.0,
            'residual_dropout': 0.0
        }
        
        # Test different layer counts
        for num_layers in [1, 2, 4]:
            config = base_config.copy()
            config['num_layers'] = num_layers
            
            model = TransformerLanguageModel(**config)
            input_ids = torch.randint(0, vocab_size, (1, 8))
            logits = model(input_ids)
            
            assert logits.shape == (1, 8, vocab_size)
            assert not torch.any(torch.isnan(logits))
    
    def test_model_numerical_stability(self, model_config):
        """Test model numerical stability"""
        model = TransformerLanguageModel(**model_config)
        
        # Test with various input patterns
        batch_size, seq_len = 2, 8
        
        # Normal input
        input_normal = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        logits_normal = model(input_normal)
        assert not torch.any(torch.isnan(logits_normal))
        assert not torch.any(torch.isinf(logits_normal))
        
        # Repeated tokens
        input_repeated = torch.full((batch_size, seq_len), 0)
        logits_repeated = model(input_repeated)
        assert not torch.any(torch.isnan(logits_repeated))
        
        # Sequential tokens
        input_sequential = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        logits_sequential = model(input_sequential)
        assert not torch.any(torch.isnan(logits_sequential))
    
    def test_model_batch_independence(self, model_config):
        """Test model processes batch samples independently"""
        model = TransformerLanguageModel(**model_config)
        model.eval()  # Set to eval mode for deterministic behavior
        
        seq_len = 8
        
        # Create different samples
        input1 = torch.randint(0, model_config['vocab_size'], (1, seq_len))
        input2 = torch.randint(0, model_config['vocab_size'], (1, seq_len))
        input_batch = torch.cat([input1, input2], dim=0)
        
        # Process batch vs individually
        logits_batch = model(input_batch)
        logits1 = model(input1)
        logits2 = model(input2)
        logits_individual = torch.cat([logits1, logits2], dim=0)
        
        assert torch.allclose(logits_batch, logits_individual, rtol=1e-6, atol=1e-6)
    
    def test_model_memory_efficiency(self, model_config):
        """Test model with larger inputs for memory efficiency"""
        model = TransformerLanguageModel(**model_config)
        
        # Test with larger batch and sequence
        batch_size = 4
        seq_len = model_config['context_length']
        input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, model_config['vocab_size'])
        assert not torch.any(torch.isnan(logits))