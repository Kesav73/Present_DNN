"""PRESENT S-box as a fixed PyTorch neural network module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SBOX = [
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2,
]


class SBoxLayer(nn.Module):
    """
    PRESENT S-box implemented as a fixed two-layer neural network using corner functions.
    
    Architecture:
    - Layer 1: 4-bit input → 16 corner detectors → ReLU
    - Layer 2: 16 corner activations → 4-bit output
    """
    
    def __init__(self, sbox: list[int] = SBOX, c: float = 0.5, device=None, dtype=torch.float32):
        """
        Initialize S-box layer with corner function weights.
        
        Args:
            sbox: 16-entry S-box lookup table
            c: Scaling parameter for corner detection
            device: Torch device (cpu or cuda)
            dtype: Data type for tensors
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        if len(sbox) != 16:
            raise ValueError("sbox must have 16 entries")
        if c <= 0:
            raise ValueError("c must be positive")
        
        # Build the two-layer network
        W1, b1 = self._build_layer1_weights(c)
        W2 = self._build_layer2_weights(sbox)
        
        # Register as buffers (not trainable)
        self.register_buffer("W1", W1)
        self.register_buffer("b1", b1)
        self.register_buffer("W2", W2)
    
    def _build_layer1_weights(self, c: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build layer 1: Corner detection for 4-bit inputs.
        
        Returns:
            (W1, b1) - weights and biases for corner detectors
        """
        W1 = torch.zeros((16, 4), dtype=self.dtype, device=self.device)
        b1 = torch.zeros(16, dtype=self.dtype, device=self.device)
        
        for corner in range(16):
            # Count bits (popcount)
            pop = sum((corner >> bit) & 1 for bit in range(4))
            
            # W1 row: ±1/c depending on bit match
            for bit in range(4):
                if (corner >> bit) & 1:
                    W1[corner, bit] = 1.0 / c
                else:
                    W1[corner, bit] = -1.0 / c
            
            # Bias for this corner
            b1[corner] = -(pop - c) / c
        
        return W1, b1
    
    def _build_layer2_weights(self, sbox: list[int]) -> torch.Tensor:
        """
        Build layer 2: Output bit construction from corner activations.
        
        Returns:
            W2 - weights mapping corners to output bits
        """
        W2 = torch.zeros((4, 16), dtype=self.dtype, device=self.device)
        
        for inp in range(16):
            out = sbox[inp]
            for bit in range(4):
                if (out >> bit) & 1:
                    W2[bit, inp] = 1.0
        
        return W2
    
    def forward(self, nibble: torch.Tensor) -> torch.Tensor:
        """
        Apply S-box DNN to 4-bit nibbles.
        
        Args:
            nibble: Binary tensor of shape (..., 4)
        
        Returns:
            S-box output of shape (..., 4)
        """
        # Handle batch dimensions
        original_shape = nibble.shape[:-1]
        nibble_flat = nibble.reshape(-1, 4)
        
        # Layer 1: Corner detection with ReLU
        h = F.linear(nibble_flat, self.W1, self.b1)  # (batch, 16)
        h = F.relu(h)
        
        # Layer 2: Output bit construction
        out = F.linear(h, self.W2)  # (batch, 4)
        
        # Round and clip to binary
        out = torch.clamp(torch.round(out), 0.0, 1.0)
        
        # Reshape back
        return out.reshape(original_shape + (4,))
