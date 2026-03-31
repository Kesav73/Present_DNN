"""80-bit PRESENT key schedule implemented with PyTorch NN primitives."""

from __future__ import annotations

import torch
import numpy as np

from ..primitives.sbox_layer_nn import SBoxLayer
from ..primitives.xor_layer_nn import XORNet
from ..utils.bit_utils import int_to_bitvec, bitvec_to_int
from ..utils.bit_utils_torch import int_to_bitvec_torch, bitvec_to_int_torch


def build_key_rotation_matrix(key_len: int = 80, rotate_by: int = 61) -> np.ndarray:
    """Build matrix for left-rotation by ``rotate_by`` over key register bits."""
    W = np.zeros((key_len, key_len), dtype=np.float32)
    for i in range(key_len):
        src = (i - rotate_by) % key_len
        W[i, src] = 1.0
    return W


def compute_all_round_keys(master_key: np.ndarray) -> list[np.ndarray]:
    """Compute K1..K32 for PRESENT-80 from an LSB-first 80-bit master key using NN primitives."""
    if master_key.shape != (80,):
        raise ValueError("master_key must have shape (80,)")

    # Initialize NN components
    xor_nn = XORNet()
    sbox_nn = SBoxLayer()
    
    # Build rotation matrix (keep as numpy for efficiency)
    W_rot = build_key_rotation_matrix()
    
    # Convert master key to torch tensor
    K = torch.from_numpy(master_key.astype(np.float32))
    round_keys: list[np.ndarray] = []

    for round_idx in range(1, 33):
        # Round key K_i uses bits k79..k16, which are indices [16:80] in LSB-first arrays.
        round_keys.append(K[16:80].detach().cpu().numpy())

        # K32 is the final whitening key. No further update is needed.
        if round_idx == 32:
            break

        # Key update for generating next round key register.
        # Rotation: convert to numpy, apply rotation, convert back
        K_np = K.detach().cpu().numpy()
        K_rotated = W_rot @ K_np
        K = torch.from_numpy(K_rotated.astype(np.float32))
        
        # Apply S-box to top nibble (bits 76-79)
        top_nibble = K[76:80].unsqueeze(0)  # Shape (1, 4)
        K[76:80] = sbox_nn(top_nibble).squeeze(0)
        
        # XOR with round counter (bits 15:20)
        counter_bits = int_to_bitvec_torch(round_idx, width=5, dtype=torch.float32)
        K[15:20] = xor_nn(K[15:20], counter_bits)

    return round_keys
