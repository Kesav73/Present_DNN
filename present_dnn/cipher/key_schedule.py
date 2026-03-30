"""80-bit PRESENT key schedule implemented with DNN-compatible primitives."""

from __future__ import annotations

import numpy as np

from ..primitives.sbox_layer import apply_sbox_dnn, build_sbox_weights
from ..primitives.xor_layer import nn_xor
from ..utils.bit_utils import int_to_bitvec


def build_key_rotation_matrix(key_len: int = 80, rotate_by: int = 61) -> np.ndarray:
    """Build matrix for left-rotation by ``rotate_by`` over key register bits."""
    W = np.zeros((key_len, key_len), dtype=np.float32)
    for i in range(key_len):
        src = (i - rotate_by) % key_len
        W[i, src] = 1.0
    return W


def compute_all_round_keys(master_key: np.ndarray) -> list[np.ndarray]:
    """Compute K1..K32 for PRESENT-80 from an LSB-first 80-bit master key."""
    if master_key.shape != (80,):
        raise ValueError("master_key must have shape (80,)")

    W_rot = build_key_rotation_matrix()
    W1, b1, W2 = build_sbox_weights()
    K = master_key.astype(np.float32).copy()
    round_keys: list[np.ndarray] = []

    for round_idx in range(1, 33):
        # Round key K_i uses bits k79..k16, which are indices [16:80] in LSB-first arrays.
        round_keys.append(K[16:80].copy())

        # K32 is the final whitening key. No further update is needed.
        if round_idx == 32:
            break

        # Key update for generating next round key register.
        K = W_rot @ K
        K[76:80] = apply_sbox_dnn(K[76:80].copy(), W1, b1, W2)
        counter_bits = int_to_bitvec(round_idx, width=5)
        K[15:20] = nn_xor(K[15:20], counter_bits)

    return round_keys
