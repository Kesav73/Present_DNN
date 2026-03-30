"""Full PRESENT-80 encryption pipeline implemented as fixed DNN operations."""

from __future__ import annotations

from ..primitives.permutation_layer import apply_player
from ..primitives.sbox_layer import apply_sbox_dnn
from ..primitives.permutation_layer import build_player_matrix
from ..primitives.sbox_layer import build_sbox_weights
from ..primitives.xor_layer import nn_xor
from ..utils.bit_utils import bitvec_to_int, int_to_bitvec
from .key_schedule import compute_all_round_keys
from .round_layer import present_round


def present_encrypt(plaintext_int: int, key_int: int, key_bits: int = 80) -> int:
    """Encrypt a 64-bit plaintext integer with PRESENT-80 key schedule."""
    if key_bits != 80:
        raise ValueError("This implementation currently supports only key_bits=80")
    if plaintext_int < 0 or plaintext_int >= (1 << 64):
        raise ValueError("plaintext_int must fit in 64 bits")
    if key_int < 0 or key_int >= (1 << 80):
        raise ValueError("key_int must fit in 80 bits")

    W_p = build_player_matrix()
    sbox_params = build_sbox_weights()
    master_key = int_to_bitvec(key_int, width=80)
    round_keys = compute_all_round_keys(master_key)

    state = int_to_bitvec(plaintext_int, width=64)
    for round_idx in range(31):
        state = present_round(state, round_keys[round_idx], W_p, sbox_params)

    state = nn_xor(state, round_keys[31])
    return bitvec_to_int(state)


def present_encrypt_with_trace(
    plaintext_int: int,
    key_int: int,
    key_bits: int = 80,
) -> tuple[int, list[dict[str, int]]]:
    """Encrypt and return detailed per-round intermediate states for debugging."""
    if key_bits != 80:
        raise ValueError("This implementation currently supports only key_bits=80")
    if plaintext_int < 0 or plaintext_int >= (1 << 64):
        raise ValueError("plaintext_int must fit in 64 bits")
    if key_int < 0 or key_int >= (1 << 80):
        raise ValueError("key_int must fit in 80 bits")

    W_p = build_player_matrix()
    W1, b1, W2 = build_sbox_weights()
    master_key = int_to_bitvec(key_int, width=80)
    round_keys = compute_all_round_keys(master_key)

    state = int_to_bitvec(plaintext_int, width=64)
    trace: list[dict[str, int]] = []

    for round_idx in range(31):
        state_after_ark = nn_xor(state, round_keys[round_idx])

        post_sbox = state_after_ark.copy()
        for i in range(16):
            start = 4 * i
            post_sbox[start : start + 4] = apply_sbox_dnn(post_sbox[start : start + 4], W1, b1, W2)

        state_after_player = apply_player(post_sbox, W_p)
        trace.append(
            {
                "round": round_idx + 1,
                "round_key": bitvec_to_int(round_keys[round_idx]),
                "after_add_round_key": bitvec_to_int(state_after_ark),
                "after_sbox": bitvec_to_int(post_sbox),
                "after_player": bitvec_to_int(state_after_player),
            }
        )
        state = state_after_player

    pre_whitening = bitvec_to_int(state)
    state = nn_xor(state, round_keys[31])
    ciphertext = bitvec_to_int(state)
    trace.append(
        {
            "round": 32,
            "round_key": bitvec_to_int(round_keys[31]),
            "before_final_whitening": pre_whitening,
            "ciphertext": ciphertext,
        }
    )
    return ciphertext, trace
