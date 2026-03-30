"""P-layer permutation as a fixed linear transform."""

from __future__ import annotations

import numpy as np


def build_player_matrix(n_bits: int = 64) -> np.ndarray:
    """Build PRESENT P-layer matrix where new_state = W @ old_state."""
    if n_bits != 64:
        raise ValueError("PRESENT P-layer is defined for 64 bits")

    W = np.zeros((n_bits, n_bits), dtype=np.float32)
    for i in range(n_bits - 1):
        dest = (16 * i) % 63
        W[dest, i] = 1.0
    W[63, 63] = 1.0
    return W


def apply_player(state: np.ndarray, W_p: np.ndarray) -> np.ndarray:
    """Apply P-layer matrix to one 64-bit state vector."""
    if state.shape != (64,):
        raise ValueError("state must have shape (64,)")
    return W_p @ state
