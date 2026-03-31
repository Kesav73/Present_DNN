"""P-layer permutation as a fixed PyTorch neural network module."""

from __future__ import annotations

import torch
import torch.nn as nn


class PermutationLayer(nn.Module):
    """
    PRESENT P-layer permutation implemented as a fixed sparse matrix.
    
    Permutation rule: P(i) = (16*i) mod 63 for i in [0, 62], P(63) = 63
    """
    
    def __init__(self, n_bits: int = 64, device=None, dtype=torch.float32):
        """
        Initialize permutation layer.
        
        Args:
            n_bits: Number of bits (must be 64 for PRESENT)
            device: Torch device (cpu or cuda)
            dtype: Data type for tensors
        """
        super().__init__()
        if n_bits != 64:
            raise ValueError("PRESENT P-layer is defined for 64 bits")
        
        self.n_bits = n_bits
        self.device = device
        self.dtype = dtype
        
        # Build permutation matrix
        W = torch.zeros((n_bits, n_bits), dtype=dtype, device=device)
        
        for i in range(n_bits - 1):
            dest = (16 * i) % 63
            W[dest, i] = 1.0
        
        W[63, 63] = 1.0
        
        # Register as buffer (not trainable)
        self.register_buffer("W", W)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply P-layer permutation.
        
        Args:
            state: Binary tensor of shape (..., 64)
        
        Returns:
            Permuted state of same shape
        """
        # Reshape to handle batch dimensions
        original_shape = state.shape
        
        if state.dim() == 1:
            # Single vector case
            return self.W @ state
        else:
            # Batch case: reshape to (batch, 64)
            batch_shape = original_shape[:-1]
            state_flat = state.reshape(-1, 64)
            
            # Apply permutation to each batch item
            result = torch.matmul(state_flat, self.W.t())
            
            # Reshape back to original shape
            return result.reshape(original_shape)
