# PRESENT Cipher as a Deep Neural Network (Implemented)

## 1) Project Status

This repository now contains a working implementation of **PRESENT-80 encryption** expressed through DNN-style fixed operations (no training):

- Bit vectors represented as float32 values in {0.0, 1.0}
- XOR implemented as ReLU composition
- S-box implemented as fixed two-layer corner-function network
- P-layer implemented as a fixed permutation matrix
- 80-bit key schedule implemented with rotation, top-nibble S-box, and round counter XOR
- Full 31-round encryption + final whitening key (K32)
- Verification script and pytest suite passing official Appendix-I vectors
- Custom CLI runner for hex plaintext and UTF-8 text input
- Optional detailed round-by-round trace output

Current scope is intentionally focused on **80-bit key mode only**.

---

## 2) What Was Implemented

## 2.1 Directory Layout

```text
present_dnn/
├── __init__.py
├── requirements.txt
├── run_custom.py
│
├── primitives/
│   ├── __init__.py
│   ├── xor_layer.py
│   ├── sbox_layer.py
│   └── permutation_layer.py
│
├── cipher/
│   ├── __init__.py
│   ├── key_schedule.py
│   ├── round_layer.py
│   └── present_dnn.py
│
├── utils/
│   ├── __init__.py
│   ├── bit_utils.py
│   └── verify.py
│
├── tests/
│   └── test_present_dnn.py
│
└── notebooks/
    └── present_dnn_walkthrough.ipynb
```

## 2.2 Module-by-Module Summary

### `present_dnn/utils/bit_utils.py`
Implemented helper conversions with LSB-first indexing:

- `int_to_bitvec(x, width)`
- `bitvec_to_int(v)`
- `hex_to_bitvec(hex_str, width=None)`

Validation and boundary checks are included for width and value ranges.

### `present_dnn/primitives/xor_layer.py`
Implemented exact bitwise XOR via ReLU identity:

$$
\mathrm{XOR}(a,b) = \mathrm{ReLU}(a-b) + \mathrm{ReLU}(b-a)
$$

Function:
- `nn_xor(a, b)`

### `present_dnn/primitives/sbox_layer.py`
Implemented PRESENT S-box as fixed DNN weights:

- `SBOX` constant: `[0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]`
- `build_sbox_weights(sbox=SBOX, c=0.5)`
- `apply_sbox_dnn(nibble, W1, b1, W2)`

Implementation details:
- Layer 1 detects each corner/input pattern.
- Layer 2 sums selected corners per output bit.
- Output is rounded/clipped to binary.

### `present_dnn/primitives/permutation_layer.py`
Implemented P-layer as matrix multiply:

- `build_player_matrix(n_bits=64)`
- `apply_player(state, W_p)`

Permutation rule used:
- `P(i) = (16 * i) mod 63` for `i in [0, 62]`
- `P(63) = 63`

### `present_dnn/cipher/key_schedule.py`
Implemented PRESENT-80 key schedule:

- `build_key_rotation_matrix(key_len=80, rotate_by=61)`
- `compute_all_round_keys(master_key)`

Behavior:
1. Extract round key `K_i` from key register bits `k79..k16` (LSB-first indices `[16:80]`).
2. If `i < 32`, update register:
   - rotate left by 61
   - apply S-box to bits `[76:80]`
   - XOR 5-bit round counter into bits `[15:20]`
3. Return `K1..K32` as 32 vectors.

### `present_dnn/cipher/round_layer.py`
Implemented one round of PRESENT:

- `present_round(state, round_key, W_p, sbox_params)`

Round steps:
1. addRoundKey (`nn_xor`)
2. sBoxLayer across 16 nibbles
3. pLayer via `W_p @ state`

### `present_dnn/cipher/present_dnn.py`
Implemented top-level encryption APIs:

- `present_encrypt(plaintext_int, key_int, key_bits=80)`
- `present_encrypt_with_trace(plaintext_int, key_int, key_bits=80)`

`present_encrypt`:
- validates key/plaintext sizes
- runs 31 rounds
- applies final whitening with `K32`
- returns ciphertext integer

`present_encrypt_with_trace`:
- returns `(ciphertext, trace)`
- trace includes per-round:
  - round key
  - state after addRoundKey
  - state after S-box
  - state after P-layer
- includes final whitening summary at round 32

### `present_dnn/utils/verify.py`
Implemented known-answer verification using 4 official vectors from Appendix I:

- `TEST_VECTORS`
- `verify_all()` returning boolean and printing PASS/FAIL lines

### `present_dnn/tests/test_present_dnn.py`
Implemented pytest coverage for:

1. int/bitvec round-trip correctness
2. Exhaustive 16-input S-box correctness
3. P-layer mapping correctness (`P(i)` formula)
4. Key schedule match against an independent integer-reference implementation
5. Known initial round keys for all-zero master key (`K1`, `K2`)
6. Official Appendix-I ciphertext vectors

### `present_dnn/run_custom.py`
Implemented custom CLI runner for practical usage:

- `--key` required (80-bit hex)
- input mode:
  - `--pt` for one 64-bit hex block
  - `--text` for UTF-8 text split into 8-byte blocks, zero-padded final block
- optional `--trace` for round-by-round state output

---

## 3) How It Works (End-to-End)

For one 64-bit block:

1. Convert plaintext int to 64-bit LSB-first vector.
2. Compute all round keys `K1..K32` from 80-bit master key.
3. Repeat 31 times:
   - XOR with round key
   - apply S-box on 16 nibbles
   - apply permutation matrix
4. Final whitening XOR with `K32`.
5. Convert final bit vector back to 64-bit integer.

No learned parameters are used. All matrices/weights are deterministic from spec logic.

---

## 4) Running the Project

## 4.1 Install Dependencies

From repository root:

```powershell
c:/Users/kunal/OneDrive/Desktop/Present/.venv/Scripts/python.exe -m pip install -r present_dnn/requirements.txt
```

## 4.2 Run Official Verification Vectors

```powershell
c:/Users/kunal/OneDrive/Desktop/Present/.venv/Scripts/python.exe -m present_dnn.utils.verify
```

Expected: all four vectors print `[PASS]`.

## 4.3 Run Tests

```powershell
c:/Users/kunal/OneDrive/Desktop/Present/.venv/Scripts/python.exe -m pytest present_dnn/tests -q
```

Current status during implementation: `6 passed`.

## 4.4 Custom Run: Hex Plaintext

```powershell
c:/Users/kunal/OneDrive/Desktop/Present/.venv/Scripts/python.exe -m present_dnn.run_custom --key 00000000000000000000 --pt 0000000000000000
```

## 4.5 Custom Run: Hex Plaintext with Step Trace

```powershell
c:/Users/kunal/OneDrive/Desktop/Present/.venv/Scripts/python.exe -m present_dnn.run_custom --key 00000000000000000000 --pt 0000000000000000 --trace
```

Trace includes `R01..R31` with `K`, `ARK`, `SB`, `PL`, then final `R32` whitening line.

## 4.6 Custom Run: Text Input

```powershell
c:/Users/kunal/OneDrive/Desktop/Present/.venv/Scripts/python.exe -m present_dnn.run_custom --key 00000000000000000000 --text "Hi"
```

Expected style of output:

```text
BLOCK 0
  PT: 4869000000000000
  CT: ...
NOTE: Text mode uses raw 8-byte blocks with zero padding and no authenticated mode.
```

Why plaintext appears as `4869000000000000`:
- `H` = `0x48`, `i` = `0x69`
- text is packed as 8-byte big-endian block
- final block is zero padded

---

## 5) Validation Results

The following correctness checks are implemented and passing:

- Official Appendix-I vectors:
  - `PT=0000000000000000, KEY=00000000000000000000 -> CT=5579c1387b228445`
  - `PT=0000000000000000, KEY=ffffffffffffffffffff -> CT=e72c46c0f5945049`
  - `PT=ffffffffffffffff, KEY=00000000000000000000 -> CT=a112ffc72f68417b`
  - `PT=ffffffffffffffff, KEY=ffffffffffffffffffff -> CT=3333dcd3213210d2`
- S-box exactness for all 16 input nibbles
- P-layer permutation mapping correctness for all bit positions
- Key schedule cross-checked against integer reference implementation
- Initial round key checks for all-zero master key (`K1=0`, `K2=c000000000000000`)

---

## 6) Design Choices and Notes

- Numeric type: `float32`
- Bit ordering: **LSB-first arrays** throughout
- XOR: ReLU-based exact identity on binary inputs
- S-box output: rounded/clipped to keep binary stability
- Key schedule update stops after producing `K32` (used for final whitening)

---

## 7) Known Scope and Limitations

- Implemented key size: **80-bit only**
- `key_bits != 80` currently raises `ValueError`
- Text mode in CLI is block conversion convenience only:
  - zero padding
  - no IV/nonce
  - no mode of operation (ECB-like per-block processing)
  - no authentication

For secure message encryption use a proper mode (for example CTR/GCM) with nonce and authentication, not raw block output.

---

## 8) Notebook

A walkthrough notebook is included:

- `present_dnn/notebooks/present_dnn_walkthrough.ipynb`

It demonstrates:
- single known-answer encryption
- selected round key display (`K1`, `K2`, `K32`)

---

## 9) References

1. Bogdanov et al., *PRESENT: An Ultra-Lightweight Block Cipher*, CHES 2007.
2. ISO/IEC 29192-2 (PRESENT standard).
3. Gerault et al., *How to Securely Implement Cryptography in Deep Neural Networks*, IACR ePrint 2025/288.

---

## 10) Next Extensions (Not Yet Implemented)

- Add PRESENT-128 key schedule support
- Add explicit decryption path
- Add mode-of-operation wrapper for text encryption demos (CTR/CBC) with proper IV handling
- Add benchmark scripts comparing integer-only vs DNN-style implementation
- Add PyTorch `nn.Module` wrappers with frozen weights
