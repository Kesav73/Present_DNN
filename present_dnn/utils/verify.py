"""Verification harness for PRESENT-DNN implementation."""

from __future__ import annotations

from ..cipher.present_dnn import present_encrypt

TEST_VECTORS = [
    (0x0000000000000000, 0x00000000000000000000, 0x5579C1387B228445),
    (0x0000000000000000, 0xFFFFFFFFFFFFFFFFFFFF, 0xE72C46C0F5945049),
    (0xFFFFFFFFFFFFFFFF, 0x00000000000000000000, 0xA112FFC72F68417B),
    (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFFFFFF, 0x3333DCD3213210D2),
]


def verify_all() -> bool:
    """Run official Appendix-I vectors and print PASS/FAIL per case."""
    all_pass = True
    for pt, key, expected_ct in TEST_VECTORS:
        ct = present_encrypt(pt, key, key_bits=80)
        ok = ct == expected_ct
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] PT={pt:016x} KEY={key:020x} -> CT={ct:016x} "
            f"(expected {expected_ct:016x})"
        )
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if verify_all() else 1)
