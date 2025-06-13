import torch
import torch.nn
from torch.optim import Optimizer

import math
from typing import Optional
from collections.abc import Callable, Iterable


class AdamW(Optimizer):
    def __init__(self, params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0.01,
                 eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get the state for this parameter
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                # update time step
                state['step'] += 1
                t = state.get('step')

                # get moment estimates and grad
                m, v = state['m'], state['v']
                grad = p.grad.data

                # Update biased first moment estimate
                m = beta1 * m + (1 - beta1) * grad
                # Update biased second moment estimate
                v = beta2 * v + (1 - beta2) * grad * grad

                # Compute bias-corrected learning rate
                adapted_lr = lr * ((1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t))

                # Update parameters
                p.data -= adapted_lr * m / (v ** 0.5 + eps)

                # Apply weight decay
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

                state['m'] = m
                state['v'] = v

        return loss


__all__ = ['AdamW']