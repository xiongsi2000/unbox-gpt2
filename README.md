# Transformer Educational Library

Minimal. Beautiful. Bold.

A hand-crafted Transformer implementation for education and research. All algorithmic logic is implemented manually, with finer granularity than typical AI-generated code. 

---

## Highlights

- **Manual BPE Tokenizer**  
  Byte Pair Encoding, implemented from scratch. No third-party magic. Understand every step. Meets basic performance requirements that gen-AI solutions cannot achieve.

- **Modular Transformer**  
  All essential components. Each file is focused, readable, and easy to extend.

- **Comprehensive Tests**  
  Every module is covered. Trust your experiments.

- **No Black Box**  
  100% human deign. Transparent, auditable, and made for deep understanding.

---

## Structure

```
transformer_educational/
├── transformer_lib/      # Core transformer modules
│   
├── data_utils/           # Data & tokenizer
│   ├── tokenizer.py      # Manual BPE
│   └── data_loader.py
│
├── examples/             # Usage examples
├── tests/                # Full test suite
└── README.md
```

---

## Why This Repo?

- **For learners**: See how every part works. No hidden logic.
- **For researchers**: Tweak, extend, and experiment with confidence.
- **For educators**: Use as a reference or teaching tool.

---

## Requirements

- PyTorch >= 1.9.0
- NumPy
- regex (for BPE)

---

## License

Educational use. Fork, modify, and explore.