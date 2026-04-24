"""Custom runner for PRESENT-DNN with optional round-by-round tracing."""

from __future__ import annotations

import argparse
from typing import Iterable

from present_dnn.cipher.present_dnn import present_encrypt, present_encrypt_with_trace


def _parse_hex(value: str, bit_width: int) -> int:
    cleaned = value.lower().strip()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("empty hex value")
    parsed = int(cleaned, 16)
    if parsed >= (1 << bit_width):
        raise ValueError(f"value exceeds {bit_width}-bit range")
    return parsed


def _text_to_blocks(text: str) -> list[int]:
    raw = text.encode("utf-8")
    if not raw:
        return [0]

    blocks: list[int] = []
    for i in range(0, len(raw), 8):
        chunk = raw[i : i + 8]
        if len(chunk) < 8:
            chunk = chunk + b"\x00" * (8 - len(chunk))
        blocks.append(int.from_bytes(chunk, byteorder="big"))
    return blocks


def _iter_block_results(
    block_values: Iterable[int],
    key_int: int,
    show_trace: bool,
):
    for idx, block in enumerate(block_values):
        if show_trace:
            ciphertext, trace = present_encrypt_with_trace(
                block,
                key_int,
                key_bits=80,
                debug_mode=True,
            )
            yield idx, block, ciphertext, trace
        else:
            ciphertext = present_encrypt(block, key_int, key_bits=80)
            yield idx, block, ciphertext, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRESENT-DNN encryption with optional trace output")
    parser.add_argument(
        "--key",
        required=True,
        help="80-bit key as hex (20 hex chars, with or without 0x)",
    )
    parser.add_argument(
        "--pt",
        help="Single 64-bit plaintext block as hex (16 hex chars)",
    )
    parser.add_argument(
        "--text",
        help="UTF-8 text to encrypt in 8-byte blocks (zero-padded final block)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print detailed per-round states (requires explicit debug-mode opt-in)",
    )

    args = parser.parse_args()

    if bool(args.pt) == bool(args.text):
        raise SystemExit("Choose exactly one input mode: --pt or --text")

    key_int = _parse_hex(args.key, bit_width=80)

    if args.pt:
        blocks = [_parse_hex(args.pt, bit_width=64)]
    else:
        blocks = _text_to_blocks(args.text)

    for block_idx, plaintext, ciphertext, trace in _iter_block_results(blocks, key_int, args.trace):
        print(f"BLOCK {block_idx}")
        print(f"  PT: {plaintext:016x}")
        print(f"  CT: {ciphertext:016x}")

        if trace is not None:
            for item in trace[:-1]:
                print(
                    "  "
                    f"R{item['round']:02d} "
                    f"K={item['round_key']:016x} "
                    f"ARK={item['after_add_round_key']:016x} "
                    f"SB={item['after_sbox']:016x} "
                    f"PL={item['after_player']:016x}"
                )
            final_item = trace[-1]
            print(
                "  "
                f"R{final_item['round']:02d} "
                f"K={final_item['round_key']:016x} "
                f"PRE={final_item['before_final_whitening']:016x} "
                f"CT={final_item['ciphertext']:016x}"
            )

    if args.text:
        print("NOTE: Text mode uses raw 8-byte blocks with zero padding and no authenticated mode.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
