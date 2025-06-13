# Transformer Educational Library

A clean Transformer implementation with a BPE tokenizer implementation. This project is written entirely by hand without any GPT, perfect for learning core Transformer concepts.

## Features

- **Manual BPE Implementation**: BPE tokenizer built from scratch, no third-party dependencies
- **Complete Transformer**: All core components included
- **Clean Code**: Well-documented with detailed comments
- **Modular Design**: Each component can be studied independently
- **Comprehensive Tests**: Extensive test suite for each component

## Project Structure

```
.
├── transformer_lib/          # Core transformer implementation
│   ├── attention.py         # Multi-head attention mechanism
│   ├── feedforward.py       # Feed-forward network
│   ├── normalization.py     # Layer normalization
│   ├── optimizers.py        # AdamW optimizer
│   ├── transformer_block.py # Transformer layer
│   ├── transformer_model.py # Complete model
│   └── utils.py            # Utility functions
├── data_utils/             # Data processing
│   ├── tokenizer.py        # Manual BPE tokenizer implementation
│   └── data_loader.py      # Data loader
└── examples/               # Example code
└── tests/                 # Comprehensive test suite
    ├── test_attention.py
    ├── test_tokenizer.py
    └── test_transformer.py
```

## Quick Start

```python
import torch
from transformer_lib import Transformer
from data_utils import Tokenizer

# Initialize model
model = Transformer(
    vocab_size=50257,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    dropout=0.1
)

# Use manually implemented BPE tokenizer
tokenizer = Tokenizer()
tokens = tokenizer.encode("Hello, world!")
```

## Why This Project?

1. **Fully Manual Implementation**: No AI tools used, code written entirely by hand
2. **BPE Tokenizer**: BPE algorithm implemented from scratch for deep understanding
3. **Educational Value**: Clear code structure and comments for learning
4. **Extensibility**: Easy to modify and experiment with

## Requirements

- PyTorch >= 1.9.0
- NumPy
- regex (for BPE tokenizer)

## License

Educational use - feel free to modify and experiment!