"""Primitive DNN layers for PRESENT."""

from .permutation_layer import apply_player, build_player_matrix
from .sbox_layer import SBOX, apply_sbox_dnn, build_sbox_weights
from .xor_layer import nn_xor

__all__ = [
    "SBOX",
    "apply_player",
    "apply_sbox_dnn",
    "build_player_matrix",
    "build_sbox_weights",
    "nn_xor",
]
