import numpy as np
import torch
from typing import Tuple


def create_data_loader(data: torch.Tensor, batch_size: int, context_length: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training batches from tokenized text data
    
    Efficiently samples random sequences from the dataset for language model training.
    Each input sequence is paired with its corresponding target sequence (shifted by 1).
    
    Args:
        data: Tokenized text data as a 1D tensor
        batch_size: Number of sequences per batch
        context_length: Length of each sequence
        device: Device to place tensors on
        
    Returns:
        Tuple of (inputs, targets) where inputs[i] and targets[i] are
        consecutive sequences for next-token prediction
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a torch.Tensor")

    data = data.to(device, non_blocking=True)
    n = data.shape[0]

    if n < batch_size * context_length:
        raise ValueError("Dataset is too small for the given batch size and context length.")

    # Generate random starting indices
    start_indices = torch.randint(0, n - context_length, (batch_size,), device=device)

    # Generate relative offsets
    offset = torch.arange(context_length, device=device)

    # Calculate indices for input sequences
    train_indices = start_indices[:, None] + offset  # (batch_size, context_length)

    # Extract input and target sequences
    inputs = data[train_indices]  # (batch_size, context_length)
    targets = data[train_indices + 1]  # Targets are shifted by 1

    return inputs, targets


def load_dataset(file_path: str, dtype=np.int32) -> np.memmap:
    """
    Load a large dataset using memory-mapped mode for efficient memory usage
    
    Args:
        file_path: Path to the .npy file containing tokenized data
        dtype: Data type of the array
        
    Returns:
        Memory-mapped numpy array
    """
    data = np.load(file_path, mmap_mode='r', allow_pickle=False)
    if data.dtype != dtype:
        data = data.astype(dtype, copy=False)
    return data