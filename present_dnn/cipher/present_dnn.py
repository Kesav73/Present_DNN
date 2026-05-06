"""Full PRESENT-80 encryption pipeline using Neural Network primitives."""

from __future__ import annotations

import torch

from ..primitives.sbox_layer_nn import SBoxLayer
from ..primitives.permutation_layer_nn import PermutationLayer
from ..utils.bit_utils import bitvec_to_int, int_to_bitvec
from ..utils.bit_utils_torch import int_to_bitvec_torch, bitvec_to_int_torch
from .key_schedule import compute_all_round_keys


def _validate_binary_tensor(value: torch.Tensor, name: str) -> None:
    """Reject any non-binary tensor values."""
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.all((value == 0.0) | (value == 1.0)):
        raise ValueError(f"{name} must be strictly binary (values in {{0.0, 1.0}})")


def _quantize_to_binary(value: torch.Tensor) -> torch.Tensor:
    """Project a tensor back onto {0.0, 1.0}."""
    quantized = torch.round(value)
    _validate_binary_tensor(quantized, "quantized state")
    return quantized


def present_encrypt(plaintext_int: int, key_int: int, key_bits: int = 80) -> int:
    """Encrypt a 64-bit plaintext integer with PRESENT-80 using NN primitives."""
    if key_bits != 80:
        raise ValueError("This implementation currently supports only key_bits=80")
    if plaintext_int < 0 or plaintext_int >= (1 << 64):
        raise ValueError("plaintext_int must fit in 64 bits")
    if key_int < 0 or key_int >= (1 << 80):
        raise ValueError("key_int must fit in 80 bits")

    # Initialize NN components
    sbox_nn = SBoxLayer()
    player_nn = PermutationLayer()
    
    # Generate round keys using original key schedule (numpy)
    master_key = int_to_bitvec(key_int, width=80)
    round_keys_np = compute_all_round_keys(master_key)
    
    # Convert plaintext to torch tensor
    state = int_to_bitvec_torch(plaintext_int, width=64, dtype=torch.float32)
    _validate_binary_tensor(state, "plaintext state")
    
    # Run 31 rounds
    for round_idx in range(31):
        # Convert round key to torch tensor
        round_key = torch.tensor(round_keys_np[round_idx], dtype=torch.float32)
        
        # Step 1: AddRoundKey (XOR with round key)
        state = torch.logical_xor(state.bool(), round_key.bool()).to(dtype=torch.float32)
        _validate_binary_tensor(state, f"round {round_idx + 1} add-round-key state")
        
        # Step 2: S-box Layer (apply to 16 nibbles)
        post_sbox = torch.zeros(64, dtype=torch.float32)
        for i in range(16):
            start = 4 * i
            nibble = state[start:start+4].unsqueeze(0)  # Shape (1, 4)
            post_sbox[start:start+4] = sbox_nn(nibble).squeeze(0)
        post_sbox = _quantize_to_binary(post_sbox)
        
        # Step 3: P-layer (bit permutation)
        state = _quantize_to_binary(player_nn(post_sbox))
    
    # Final whitening: XOR with K32
    round_key_32 = torch.tensor(round_keys_np[31], dtype=torch.float32)
    state = torch.logical_xor(state.bool(), round_key_32.bool()).to(dtype=torch.float32)
    _validate_binary_tensor(state, "ciphertext state")
    
    # Convert back to integer
    return bitvec_to_int_torch(state)



def present_encrypt_with_trace(
    plaintext_int: int,
    key_int: int,
    key_bits: int = 80,
    debug_mode: bool = False,
) -> tuple[int, list[dict[str, int]]]:
    """Encrypt and return detailed per-round intermediate states using NN primitives."""
    if not debug_mode:
        raise ValueError("trace output is disabled unless debug_mode=True")
    if key_bits != 80:
        raise ValueError("This implementation currently supports only key_bits=80")
    if plaintext_int < 0 or plaintext_int >= (1 << 64):
        raise ValueError("plaintext_int must fit in 64 bits")
    if key_int < 0 or key_int >= (1 << 80):
        raise ValueError("key_int must fit in 80 bits")

    # Initialize NN components
    sbox_nn = SBoxLayer()
    player_nn = PermutationLayer()
    
    # Generate round keys using original key schedule (numpy)
    master_key = int_to_bitvec(key_int, width=80)
    round_keys_np = compute_all_round_keys(master_key)

    # Convert plaintext to torch tensor
    state = int_to_bitvec_torch(plaintext_int, width=64, dtype=torch.float32)
    _validate_binary_tensor(state, "plaintext state")
    
    trace: list[dict[str, int]] = []

    # Run 31 rounds
    for round_idx in range(31):
        # Convert round key to torch tensor
        round_key = torch.tensor(round_keys_np[round_idx], dtype=torch.float32)
        
        # Step 1: AddRoundKey (XOR with round key)
        state_after_ark = torch.logical_xor(state.bool(), round_key.bool()).to(dtype=torch.float32)
        _validate_binary_tensor(state_after_ark, f"round {round_idx + 1} add-round-key state")
        
        # Step 2: S-box Layer (apply to 16 nibbles)
        post_sbox = torch.zeros(64, dtype=torch.float32)
        for i in range(16):
            start = 4 * i
            nibble = state_after_ark[start:start+4].unsqueeze(0)  # Shape (1, 4)
            post_sbox[start:start+4] = sbox_nn(nibble).squeeze(0)
        post_sbox = _quantize_to_binary(post_sbox)
        
        # Step 3: P-layer (bit permutation)
        state_after_player = _quantize_to_binary(player_nn(post_sbox))
        
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
    state = torch.logical_xor(state.bool(), round_key_32.bool()).to(dtype=torch.float32)
    _validate_binary_tensor(state, "ciphertext state")
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
