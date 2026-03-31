#!/usr/bin/env python3
"""Simple test runner for all tests."""

import sys
sys.path.insert(0, '/home/kesav/Desktop')

import torch
import numpy as np

from present_dnn.cipher.key_schedule import compute_all_round_keys
from present_dnn.cipher.present_dnn import present_encrypt
from present_dnn.primitives.sbox_layer_nn import SBoxLayer, SBOX
from present_dnn.primitives.permutation_layer_nn import PermutationLayer
from present_dnn.utils.bit_utils import bitvec_to_int, int_to_bitvec
from present_dnn.utils.bit_utils_torch import int_to_bitvec_torch, bitvec_to_int_torch
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
    print("Testing int_to_bitvec roundtrip...", end=" ")
    samples = [0, 1, 0xABCD, (1 << 63), (1 << 64) - 1]
    for value in samples:
        vec = int_to_bitvec(value, width=64)
        assert bitvec_to_int(vec) == value
    print("✓")


def test_sbox_exact_for_all_nibbles():
    print("Testing S-box for all nibbles...", end=" ")
    sbox_nn = SBoxLayer()
    for nibble in range(16):
        nibble_torch = int_to_bitvec_torch(nibble, width=4, dtype=torch.float32).unsqueeze(0)
        out_torch = sbox_nn(nibble_torch).squeeze(0)
        assert bitvec_to_int_torch(out_torch) == SBOX[nibble]
    print("✓")


def test_player_mapping_matches_spec_formula():
    print("Testing P-layer mapping...", end=" ")
    player = PermutationLayer()
    W = player.W
    for i in range(63):
        src = torch.zeros(64, dtype=torch.float32)
        src[i] = 1.0
        dst = W @ src
        assert int(torch.argmax(dst)) == (16 * i) % 63
    src = torch.zeros(64, dtype=torch.float32)
    src[63] = 1.0
    dst = W @ src
    assert int(torch.argmax(dst)) == 63
    print("✓")


def test_key_schedule_matches_integer_reference():
    print("Testing key schedule...", end=" ")
    for mk in [0x00000000000000000000, 0xFFFFFFFFFFFFFFFFFFFF, 0x0123456789ABCDEF0123]:
        ref_keys = _ref_key_schedule_80(mk)
        dnn_keys = compute_all_round_keys(int_to_bitvec(mk, width=80))
        assert len(dnn_keys) == 32
        assert [bitvec_to_int(k) for k in dnn_keys] == ref_keys
    print("✓")


def test_known_first_round_keys_zero_master_key():
    print("Testing first round keys (zero key)...", end=" ")
    keys = compute_all_round_keys(int_to_bitvec(0, width=80))
    assert bitvec_to_int(keys[0]) == 0x0000000000000000
    assert bitvec_to_int(keys[1]) == 0xC000000000000000
    print("✓")


def test_appendix_vectors():
    print("Testing known test vectors...", end=" ")
    for pt, key, expected in TEST_VECTORS:
        result = present_encrypt(pt, key, key_bits=80)
        print(f"\n  PT={pt:016x}, KEY={key:020x} -> {result:016x} (expected {expected:016x})", end="")
        assert result == expected, f"Mismatch: got {result:016x}, expected {expected:016x}"
    print(" ✓")


if __name__ == "__main__":
    try:
        test_int_bitvec_roundtrip()
        test_sbox_exact_for_all_nibbles()
        test_player_mapping_matches_spec_formula()
        test_key_schedule_matches_integer_reference()
        test_known_first_round_keys_zero_master_key()
        test_appendix_vectors()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
