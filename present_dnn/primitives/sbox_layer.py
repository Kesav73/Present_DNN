"""PRESENT S-box as a fixed two-layer DNN."""

from __future__ import annotations

import numpy as np

SBOX = [
    0xC,
    0x5,
    0x6,
    0xB,
    0x9,
    0x0,
    0xA,
    0xD,
    0x3,
    0xE,
    0xF,
    0x8,
    0x4,
    0x7,
    0x1,
    0x2,
]


def build_sbox_weights(sbox: list[int] = SBOX, c: float = 0.5):
    """Build fixed weights for corner-function based 4-bit S-box mapping."""
    if len(sbox) != 16:
        raise ValueError("sbox must have 16 entries")
    if c <= 0:
        raise ValueError("c must be positive")

    W1 = np.zeros((16, 4), dtype=np.float32)
    b1 = np.zeros(16, dtype=np.float32)
    W2 = np.zeros((4, 16), dtype=np.float32)

    for corner in range(16):
        pop = (corner >> 0 & 1) + (corner >> 1 & 1) + (corner >> 2 & 1) + (corner >> 3 & 1)
        for bit in range(4):
            W1[corner, bit] = (1.0 / c) if ((corner >> bit) & 1) else (-1.0 / c)
        b1[corner] = -(pop - c) / c

    for inp in range(16):
        out = sbox[inp]
        for bit in range(4):
            if (out >> bit) & 1:
                W2[bit, inp] = 1.0

    return W1, b1, W2


def apply_sbox_dnn(nibble: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Apply fixed S-box DNN to one 4-bit LSB-first nibble."""
    if nibble.shape != (4,):
        raise ValueError("nibble must have shape (4,)")
    h = np.maximum(W1 @ nibble + b1, 0.0)
    out = W2 @ h
    return np.clip(np.round(out), 0.0, 1.0).astype(np.float32)
