"""Bit-vector conversion helpers for PyTorch using LSB-first indexing."""

from __future__ import annotations

import torch


def int_to_bitvec_torch(x: int, width: int = 64, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a float32 bit vector of length ``width`` with LSB at index 0."""
    if width <= 0:
        raise ValueError("width must be positive")
    if x < 0:
        raise ValueError("x must be non-negative")
    
    bits = [(x >> i) & 1 for i in range(width)]
    return torch.tensor(bits, dtype=dtype, device=device)


def bitvec_to_int_torch(v: torch.Tensor) -> int:
    """Convert an LSB-first bit vector into a Python integer."""
    v_np = v.detach().cpu().numpy() if v.is_cuda or v.requires_grad else v.numpy()
    return int(sum((int(round(float(b))) & 1) << i for i, b in enumerate(v_np)))


def hex_to_bitvec_torch(hex_str: str, width: int | None = None, device=None, dtype=torch.float32) -> torch.Tensor:
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
    return int_to_bitvec_torch(value, width=final_width, device=device, dtype=dtype)
