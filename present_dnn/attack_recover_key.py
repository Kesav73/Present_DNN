"""Differential key-recovery demo attack for PRESENT-DNN.

This script demonstrates the technique documented in attacks.md:
1) Recover first round key K1 (64 bits) from epsilon perturbations
2) Brute-force remaining 16 master-key bits

Run from project root:
    python -m present_dnn.attack_recover_key --key 0123456789abcdef0123
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable

import torch

from present_dnn.cipher.key_schedule import compute_all_round_keys
from present_dnn.cipher.present_dnn import present_encrypt
from present_dnn.primitives.xor_layer_nn import XORNet
from present_dnn.utils.bit_utils import int_to_bitvec, bitvec_to_int


VectorOracle = Callable[[torch.Tensor], torch.Tensor]
EncryptOracle = Callable[[int], int]


@dataclass(frozen=True)
class AttackResult:
    recovered_k1: int
    recovered_master_key: int | None
    candidates_checked: int
    verification_pairs: int


def _parse_hex(value: str, bit_width: int) -> int:
    cleaned = value.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("empty hex value")
    parsed = int(cleaned, 16)
    if parsed >= (1 << bit_width):
        raise ValueError(f"value exceeds {bit_width}-bit range")
    return parsed


def _validate_binary_tensor(value: torch.Tensor, name: str) -> None:
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.all((value == 0.0) | (value == 1.0)):
        raise ValueError(f"{name} must be strictly binary (values in {{0.0, 1.0}})")


def _build_first_round_oracle(secret_key: int) -> VectorOracle:
    """Return oracle for first AddRoundKey output on real-valued plaintext vectors.

    This matches the vulnerable step: y = XOR(x, K1) = |x - K1| elementwise.
    """
    if secret_key < 0 or secret_key >= (1 << 80):
        raise ValueError("secret_key must be 80-bit")

    round_keys = compute_all_round_keys(int_to_bitvec(secret_key, width=80))
    k1 = torch.tensor(round_keys[0], dtype=torch.float32)
    xor_nn = XORNet()

    def oracle(plaintext_vec: torch.Tensor) -> torch.Tensor:
        if plaintext_vec.shape != (64,):
            raise ValueError("plaintext_vec must be shape (64,)")
        plaintext_vec = plaintext_vec.to(dtype=torch.float32)
        _validate_binary_tensor(plaintext_vec, "plaintext_vec")
        return xor_nn(plaintext_vec, k1)

    return oracle


def _build_encrypt_oracle(secret_key: int) -> EncryptOracle:
    def oracle(plaintext_int: int) -> int:
        return present_encrypt(plaintext_int, secret_key, key_bits=80)

    return oracle


def recover_k1_from_epsilon_test(
    first_round_oracle: VectorOracle,
    epsilon: float,
    threshold: float,
    seed: int,
) -> int:
    """Recover 64-bit K1 using C+ vs C- perturbation decisions per bit."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    rng = random.Random(seed)
    base_bits = [rng.randint(0, 1) for _ in range(64)]
    base_vec = torch.tensor(base_bits, dtype=torch.float32)

    recovered_bits = [0] * 64

    for bit_idx in range(64):
        p_plus = base_vec.clone()
        p_minus = base_vec.clone()
        p_plus[bit_idx] += epsilon
        p_minus[bit_idx] -= epsilon

        y_plus = first_round_oracle(p_plus)
        y_minus = first_round_oracle(p_minus)

        distance = float(torch.abs(y_plus - y_minus).sum().item())

        if distance <= threshold:
            recovered_bits[bit_idx] = base_bits[bit_idx]
        else:
            recovered_bits[bit_idx] = 1 - base_bits[bit_idx]

    return bitvec_to_int(recovered_bits)


def brute_force_master_key_from_k1(
    recovered_k1: int,
    verification_pairs: list[tuple[int, int]],
) -> tuple[int | None, int]:
    """Brute-force unknown low 16 bits of 80-bit key from recovered K1."""
    checked = 0

    for low16 in range(1 << 16):
        candidate_key = (recovered_k1 << 16) | low16
        checked += 1

        ok = True
        for pt, expected_ct in verification_pairs:
            if present_encrypt(pt, candidate_key, key_bits=80) != expected_ct:
                ok = False
                break

        if ok:
            return candidate_key, checked

    return None, checked


def run_attack(
    secret_key: int,
    epsilon: float,
    threshold: float,
    seed: int,
    pair_count: int,
) -> AttackResult:
    first_round_oracle = _build_first_round_oracle(secret_key)
    encrypt_oracle = _build_encrypt_oracle(secret_key)

    recovered_k1 = recover_k1_from_epsilon_test(
        first_round_oracle=first_round_oracle,
        epsilon=epsilon,
        threshold=threshold,
        seed=seed,
    )

    rng = random.Random(seed + 1)
    verification_pairs: list[tuple[int, int]] = []
    for _ in range(pair_count):
        pt = rng.getrandbits(64)
        ct = encrypt_oracle(pt)
        verification_pairs.append((pt, ct))

    recovered_master_key, checked = brute_force_master_key_from_k1(
        recovered_k1=recovered_k1,
        verification_pairs=verification_pairs,
    )

    return AttackResult(
        recovered_k1=recovered_k1,
        recovered_master_key=recovered_master_key,
        candidates_checked=checked,
        verification_pairs=pair_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover PRESENT-80 key using epsilon perturbation attack")
    parser.add_argument("--key", required=True, help="Secret 80-bit key as hex (20 hex chars)")
    parser.add_argument("--eps", type=float, default=0.01, help="Perturbation epsilon")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Distance threshold for bit decision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for base plaintext and pairs")
    parser.add_argument("--pairs", type=int, default=3, help="Plaintext/ciphertext pairs for low-16 brute-force")

    args = parser.parse_args()

    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")

    secret_key = _parse_hex(args.key, bit_width=80)

    result = run_attack(
        secret_key=secret_key,
        epsilon=args.eps,
        threshold=args.threshold,
        seed=args.seed,
        pair_count=args.pairs,
    )

    true_k1 = bitvec_to_int(compute_all_round_keys(int_to_bitvec(secret_key, width=80))[0])

    print("=== PRESENT-DNN Key-Recovery Attack Demo ===")
    print(f"Secret key       : {secret_key:020x}")
    print(f"Recovered K1     : {result.recovered_k1:016x}")
    print(f"True K1          : {true_k1:016x}")
    print(f"K1 match         : {result.recovered_k1 == true_k1}")
    print(f"Bruteforce checked: {result.candidates_checked}")
    print(f"Pairs used       : {result.verification_pairs}")

    if result.recovered_master_key is None:
        print("Recovered key    : <not found>")
        return 1

    print(f"Recovered key    : {result.recovered_master_key:020x}")
    print(f"Master key match : {result.recovered_master_key == secret_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
