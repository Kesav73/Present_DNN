"""XOR primitive expressed with ReLU operations."""

from __future__ import annotations

import numpy as np


def nn_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute elementwise XOR for binary vectors via ReLU identity."""
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shapes")
    return np.maximum(a - b, 0.0) + np.maximum(b - a, 0.0)
