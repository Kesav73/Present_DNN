"""Full PRESENT-80 encryption pipeline using Neural Network primitives."""

from __future__ import annotations

import torch
import numpy as np

from ..primitives.xor_layer_nn import XORNet
from ..primitives.sbox_layer_nn import SBoxLayer
from ..primitives.permutation_layer_nn import PermutationLayer
from ..utils.bit_utils import bitvec_to_int, int_to_bitvec
from ..utils.bit_utils_torch import int_to_bitvec_torch, bitvec_to_int_torch
from .key_schedule import compute_all_round_keys


def present_encrypt(plaintext_int: int, key_int: int, key_bits: int = 80) -> int:
    """Encrypt a 64-bit plaintext integer with PRESENT-80 using NN primitives."""
    if key_bits != 80:
        raise ValueError("This implementation currently supports only key_bits=80")
    if plaintext_int < 0 or plaintext_int >= (1 << 64):
        raise ValueError("plaintext_int must fit in 64 bits")
    if key_int < 0 or key_int >= (1 << 80):
        raise ValueError("key_int must fit in 80 bits")

    # Initialize NN components
    xor_nn = XORNet()
    sbox_nn = SBoxLayer()
    player_nn = PermutationLayer()
    
    # Generate round keys using original key schedule (numpy)
    master_key = int_to_bitvec(key_int, width=80)
    round_keys_np = compute_all_round_keys(master_key)
    
    # Convert plaintext to torch tensor
    state = int_to_bitvec_torch(plaintext_int, width=64, dtype=torch.float32)
    
    # Run 31 rounds
    for round_idx in range(31):
        # Convert round key to torch tensor
        round_key = torch.tensor(round_keys_np[round_idx], dtype=torch.float32)
        
        # Step 1: AddRoundKey (XOR with round key)
        state = xor_nn(state, round_key)
        
        # Step 2: S-box Layer (apply to 16 nibbles)
        post_sbox = torch.zeros(64, dtype=torch.float32)
        for i in range(16):
            start = 4 * i
            nibble = state[start:start+4].unsqueeze(0)  # Shape (1, 4)
            post_sbox[start:start+4] = sbox_nn(nibble).squeeze(0)
        
        # Step 3: P-layer (bit permutation)
        state = player_nn(post_sbox)
    
    # Final whitening: XOR with K32
    round_key_32 = torch.tensor(round_keys_np[31], dtype=torch.float32)
    state = xor_nn(state, round_key_32)
    
    # Convert back to integer
    return bitvec_to_int_torch(state)



def present_encrypt_with_trace(
    plaintext_int: int,
    key_int: int,
    key_bits: int = 80,
) -> tuple[int, list[dict[str, int]]]:
    """Encrypt and return detailed per-round intermediate states using NN primitives."""
    if key_bits != 80:
        raise ValueError("This implementation currently supports only key_bits=80")
    if plaintext_int < 0 or plaintext_int >= (1 << 64):
        raise ValueError("plaintext_int must fit in 64 bits")
    if key_int < 0 or key_int >= (1 << 80):
        raise ValueError("key_int must fit in 80 bits")

    # Initialize NN components
    xor_nn = XORNet()
    sbox_nn = SBoxLayer()
    player_nn = PermutationLayer()
    
    # Generate round keys using original key schedule (numpy)
    master_key = int_to_bitvec(key_int, width=80)
    round_keys_np = compute_all_round_keys(master_key)

    # Convert plaintext to torch tensor
    state = int_to_bitvec_torch(plaintext_int, width=64, dtype=torch.float32)
    
    trace: list[dict[str, int]] = []

    # Run 31 rounds
    for round_idx in range(31):
        # Convert round key to torch tensor
        round_key = torch.tensor(round_keys_np[round_idx], dtype=torch.float32)
        
        # Step 1: AddRoundKey (XOR with round key)
        state_after_ark = xor_nn(state, round_key)
        
        # Step 2: S-box Layer (apply to 16 nibbles)
        post_sbox = torch.zeros(64, dtype=torch.float32)
        for i in range(16):
            start = 4 * i
            nibble = state_after_ark[start:start+4].unsqueeze(0)  # Shape (1, 4)
            post_sbox[start:start+4] = sbox_nn(nibble).squeeze(0)
        
        # Step 3: P-layer (bit permutation)
        state_after_player = player_nn(post_sbox)
        
        # Record trace
        trace.append(
            {
                "round": round_idx + 1,
                "round_key": bitvec_to_int(round_keys_np[round_idx]),
                "after_add_round_key": bitvec_to_int_torch(state_after_ark),
                "after_sbox": bitvec_to_int_torch(post_sbox),
                "after_player": bitvec_to_int_torch(state_after_player),
            }
        )
        state = state_after_player

    # Final whitening: XOR with K32
    pre_whitening = bitvec_to_int_torch(state)
    round_key_32 = torch.tensor(round_keys_np[31], dtype=torch.float32)
    state = xor_nn(state, round_key_32)
    ciphertext = bitvec_to_int_torch(state)
    
    trace.append(
        {
            "round": 32,
            "round_key": bitvec_to_int(round_keys_np[31]),
            "before_final_whitening": pre_whitening,
            "ciphertext": ciphertext,
        }
    )
    return ciphertext, trace
