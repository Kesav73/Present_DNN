"""Bit-vector conversion helpers using LSB-first indexing."""

from __future__ import annotations

import numpy as np


def int_to_bitvec(x: int, width: int = 64) -> np.ndarray:
    """Return a float32 bit vector of length ``width`` with LSB at index 0."""
    if width <= 0:
        raise ValueError("width must be positive")
    if x < 0:
        raise ValueError("x must be non-negative")
    return np.array([(x >> i) & 1 for i in range(width)], dtype=np.float32)


def bitvec_to_int(v: np.ndarray) -> int:
    """Convert an LSB-first bit vector into a Python integer."""
    return int(sum((int(round(float(b))) & 1) << i for i, b in enumerate(v)))


def hex_to_bitvec(hex_str: str, width: int | None = None) -> np.ndarray:
    """Convert a hex string into an LSB-first float32 bit vector."""
    cleaned = hex_str.lower().strip()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("hex string is empty")

    inferred_width = len(cleaned) * 4
    final_width = inferred_width if width is None else width
    value = int(cleaned, 16)
    if value >= (1 << final_width):
        raise ValueError("hex value does not fit requested width")
    return int_to_bitvec(value, width=final_width)
