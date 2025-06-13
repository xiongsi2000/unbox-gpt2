"""
Simple Training Example for Transformer Language Model

This example demonstrates how to train a small transformer model
on sample text data for educational purposes.
"""

import torch
import torch.nn as nn
from transformer_lib import TransformerLanguageModel, AdamW
from data_utils.data_loader import create_data_loader


def train_model():
    """Train a simple transformer language model"""
    
    # Model configuration
    config = {
        'vocab_size': 1000,
        'context_length': 128,
        'd_model': 256,
        'num_layers': 4,
        'num_heads': 8,
        'd_ff': 1024,
        'attn_dropout': 0.1,
        'residual_dropout': 0.1
    }
    
    # Training configuration
    batch_size = 16
    learning_rate = 3e-4
    num_steps = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = TransformerLanguageModel(**config)
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy training data (replace with real tokenized data)
    dummy_data = torch.randint(0, config['vocab_size'], (10000,))
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    for step in range(num_steps):
        # Get batch
        inputs, targets = create_data_loader(
            dummy_data, batch_size, config['context_length'], device
        )
        
        # Forward pass
        logits = model(inputs)
        
        # Calculate loss
        loss = criterion(
            logits.view(-1, config['vocab_size']), 
            targets.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    print("Training completed!")
    return model


def generate_text(model, tokenizer=None, prompt="Hello", max_length=50):
    """Generate text using the trained model"""
    model.eval()
    
    # For demonstration, use simple character-level tokenization
    if tokenizer is None:
        # Simple character tokenization (for demo only)
        vocab = list(set(prompt))
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        idx_to_char = {i: ch for i, ch in enumerate(vocab)}
        
        # Convert prompt to token IDs
        input_ids = torch.tensor([char_to_idx.get(ch, 0) for ch in prompt]).unsqueeze(0)
    else:
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_length=max_length, 
            temperature=0.8,
            top_k=40
        )
    
    if tokenizer is None:
        # Convert back to text (demo only)
        generated_text = ''.join([idx_to_char.get(idx.item(), '?') for idx in generated[0]])
        return generated_text
    else:
        return tokenizer.decode(generated[0].tolist())


if __name__ == "__main__":
    # Train the model
    trained_model = train_model()
    
    # Generate some text (basic demo)
    print("\nGenerating text...")
    generated = generate_text(trained_model, prompt="The")
    print(f"Generated: {generated}")