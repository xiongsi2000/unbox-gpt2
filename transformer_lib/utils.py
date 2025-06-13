import torch
import os
from typing import Dict, Any


def save_checkpoint(model, optimizer, epoch=None, loss=None, path=None, iteration=None, out=None):
    """Save a training checkpoint"""
    # Support both calling conventions
    if out is not None:
        path = out
    if iteration is not None:
        epoch = iteration
        
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if iteration is not None:
        checkpoint['iteration'] = iteration  
    if loss is not None:
        checkpoint['loss'] = loss
        
    torch.save(checkpoint, path)


def load_checkpoint(path=None, model=None, optimizer=None, checkpoint_path=None):
    """Load a training checkpoint"""
    # Support both calling conventions
    if checkpoint_path is not None:
        path = checkpoint_path
        
    checkpoint = torch.load(path)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return iteration if available, otherwise epoch, otherwise 0
    if 'iteration' in checkpoint:
        return checkpoint['iteration']
    elif 'epoch' in checkpoint:
        return checkpoint['epoch'] 
    else:
        return 0


__all__ = ['save_checkpoint', 'load_checkpoint']