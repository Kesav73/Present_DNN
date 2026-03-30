"""Pytest coverage for PRESENT-DNN implementation."""

from __future__ import annotations

import numpy as np

from present_dnn.cipher.key_schedule import compute_all_round_keys
from present_dnn.cipher.present_dnn import present_encrypt
from present_dnn.primitives.permutation_layer import build_player_matrix
from present_dnn.primitives.sbox_layer import SBOX, apply_sbox_dnn, build_sbox_weights
from present_dnn.utils.bit_utils import bitvec_to_int, int_to_bitvec
from present_dnn.utils.verify import TEST_VECTORS


def _ref_key_schedule_80(master_key: int) -> list[int]:
    mask80 = (1 << 80) - 1
    key_reg = master_key & mask80
    keys: list[int] = []

    for round_idx in range(1, 33):
        keys.append((key_reg >> 16) & ((1 << 64) - 1))
        if round_idx == 32:
            break

        key_reg = ((key_reg << 61) & mask80) | (key_reg >> 19)

        top_nibble = (key_reg >> 76) & 0xF
        key_reg &= ~((0xF) << 76)
        key_reg |= SBOX[top_nibble] << 76

        key_reg ^= (round_idx & 0x1F) << 15

    return keys


def test_int_bitvec_roundtrip():
    samples = [0, 1, 0xABCD, (1 << 63), (1 << 64) - 1]
    for value in samples:
        vec = int_to_bitvec(value, width=64)
        assert bitvec_to_int(vec) == value


def test_sbox_exact_for_all_nibbles():
    W1, b1, W2 = build_sbox_weights()
    for nibble in range(16):
        nibble_vec = int_to_bitvec(nibble, width=4)
        out_vec = apply_sbox_dnn(nibble_vec, W1, b1, W2)
        assert bitvec_to_int(out_vec) == SBOX[nibble]


def test_player_mapping_matches_spec_formula():
    W = build_player_matrix()
    for i in range(63):
        src = np.zeros(64, dtype=np.float32)
        src[i] = 1.0
        dst = W @ src
        assert int(np.argmax(dst)) == (16 * i) % 63
    src = np.zeros(64, dtype=np.float32)
    src[63] = 1.0
    dst = W @ src
    assert int(np.argmax(dst)) == 63


def test_key_schedule_matches_integer_reference():
    for mk in [0x00000000000000000000, 0xFFFFFFFFFFFFFFFFFFFF, 0x0123456789ABCDEF0123]:
        ref_keys = _ref_key_schedule_80(mk)
        dnn_keys = compute_all_round_keys(int_to_bitvec(mk, width=80))
        assert len(dnn_keys) == 32
        assert [bitvec_to_int(k) for k in dnn_keys] == ref_keys


def test_known_first_round_keys_zero_master_key():
    keys = compute_all_round_keys(int_to_bitvec(0, width=80))
    assert bitvec_to_int(keys[0]) == 0x0000000000000000
    assert bitvec_to_int(keys[1]) == 0xC000000000000000


def test_appendix_vectors():
    for pt, key, expected in TEST_VECTORS:
        assert present_encrypt(pt, key, key_bits=80) == expected
