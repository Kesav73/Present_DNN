"""Single PRESENT round as composition of DNN primitives."""

from __future__ import annotations

import numpy as np

from ..primitives.permutation_layer import apply_player
from ..primitives.sbox_layer import apply_sbox_dnn
from ..primitives.xor_layer import nn_xor


def present_round(
    state: np.ndarray,
    round_key: np.ndarray,
    W_p: np.ndarray,
    sbox_params: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Apply one PRESENT round: addRoundKey, sBoxLayer, pLayer."""
    if state.shape != (64,):
        raise ValueError("state must have shape (64,)")
    if round_key.shape != (64,):
        raise ValueError("round_key must have shape (64,)")

    W1, b1, W2 = sbox_params

    state = nn_xor(state, round_key)
    post_sbox = np.zeros(64, dtype=np.float32)
    for i in range(16):
        start = 4 * i
        post_sbox[start : start + 4] = apply_sbox_dnn(state[start : start + 4], W1, b1, W2)

    return apply_player(post_sbox, W_p)
