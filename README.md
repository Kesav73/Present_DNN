# PRESENT Cipher as a Deep Neural Network (PyTorch Implementation)

## 1) Project Status

This repository contains a **fully working PRESENT-80 encryption** implementation using **PyTorch neural networks** with no learnable parameters—all operations are fixed, deterministic transforms:

### Core Features
- **XORNet**: XOR operation as a 2-layer ReLU network (fixed hardcoded weights)
- **SBoxLayer**: 4-bit S-box as a corner-function 2-layer neural network
- **PermutationLayer**: 64-bit P-layer as a sparse permutation matrix
- **Key Schedule**: NumPy-based with internal use of NN primitives for S-box and XOR
- **Full Encryption**: 31 rounds + final whitening key (K32)
- **Bit Format**: Float32 tensors in {0.0, 1.0} with LSB-first indexing
- **Verification**: All 4 official test vectors passing
- **Custom CLI**: Hex plaintext, UTF-8 text input, optional trace output
- **Test Suite**: Comprehensive pytest coverage with custom test runner

**Scope**: 80-bit key mode only (PRESENT-80)

---

## 2) What Was Implemented

## 2.1 Directory Layout

```text
present_dnn/
├── __init__.py
├── requirements.txt
├── run_custom.py
│
├── primitives/                    # PyTorch NN implementations
│   ├── __init__.py
│   ├── xor_layer_nn.py           # XORNet: 2-layer ReLU XOR
│   ├── sbox_layer_nn.py          # SBoxLayer: corner-function S-box
│   └── permutation_layer_nn.py   # PermutationLayer: sparse matrix
│
├── cipher/
│   ├── __init__.py
│   ├── key_schedule.py           # NumPy-based with NN primitives internally
│   └── present_dnn.py            # Full 31-round encryption pipeline
│
├── utils/
│   ├── __init__.py
│   ├── bit_utils.py              # NumPy bit conversions
│   ├── bit_utils_torch.py        # PyTorch bit conversions
│   └── verify.py                 # Official test vectors
│
├── tests/
│   ├── test_present_dnn.py       # Pytest suite
│   └── run_tests.py              # Custom test runner
│
└── notebooks/
    └── present_dnn_walkthrough.ipynb
```

## 2.2 Module-by-Module Summary

### `present_dnn/utils/bit_utils.py`
NumPy-based helper conversions with LSB-first indexing:

- `int_to_bitvec(x, width)`: Integer to bit vector (NumPy array)
- `bitvec_to_int(v)`: Bit vector to integer
- `hex_to_bitvec(hex_str, width=None)`: Hex string to bit vector

Validation and boundary checks included for width and value ranges.

### `present_dnn/utils/bit_utils_torch.py` (NEW)
PyTorch-based equivalents for tensor operation:

- `int_to_bitvec_torch(x, width, dtype=torch.float32, device='cpu')`
- `bitvec_to_int_torch(v)`: Tensor to integer
- `hex_to_bitvec_torch(hex_str, width=None, dtype=torch.float32)`

Used internally by NN primitives for tensor conversions.

### `present_dnn/primitives/xor_layer_nn.py`
**PyTorch neural network implementing XOR** via ReLU identity:

$$
\mathrm{XOR}(a,b) = \mathrm{ReLU}(a-b) + \mathrm{ReLU}(b-a)
$$

Class:
- `XORNet(input_size=2)`: 2 hidden neurons (ReLU) with hardcoded weights `W1=[[1,-1],[-1,1]]` and `W2=[1,1]`
- Forward pass: `(inputs) -> hidden (ReLU) -> output`
- Supports batched operation with shape `(batch_size, input_size)`

### `present_dnn/primitives/sbox_layer_nn.py`
**PyTorch neural network implementing PRESENT S-box** as a corner-function 2-layer network:

- `SBOX` constant: `[0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]`
- `SBoxLayer()`: Takes 4-bit nibble input, outputs 4-bit S-box result

Implementation details:
- **Layer 1**: 16 neurons detect which of 16 corners the input matches (ReLU activation)
- **Layer 2**: 4 output neurons sum selected corners to compute each output bit
- Fixed, non-trainable weights derived from S-box truth table
- Supports batched operation with shape `(batch_size, 4)`

### `present_dnn/primitives/permutation_layer_nn.py`
**PyTorch implementation of P-layer** as a sparse permutation matrix:

- `PermutationLayer()`: Takes 64-bit state, outputs permuted 64-bit state
- Permutation rule: `P(i) = (16 * i) mod 63` for `i in [0, 62]`, `P(63) = 63`
- Implemented as 64×64 sparse matrix (one 1.0 per column, rest zeros)
- Forward pass: dense matrix-vector multiplication `W @ state`
- Non-trainable weights, supports batched operation with shape `(batch_size, 64)`

### `present_dnn/cipher/key_schedule.py`
**PRESENT-80 key schedule** using NN primitives internally:

- `compute_all_round_keys(master_key)`: Takes 80-bit master key, returns 32 round keys

Behavior:
1. Extract round key `K_i` from key register bits `k79..k16` (LSB-first indices `[16:80]`)
2. If `i < 32`, update register:
   - rotate left by 61
   - apply `SBoxLayer` (NN) to bits `[76:80]`
   - XOR 5-bit round counter into bits `[15:20]` using `XORNet` (NN)
3. Return `K1..K32` as 32 bit-vector arrays

**Note**: Uses PyTorch NN primitives internally (SBoxLayer, XORNet) but maintains NumPy-compatible interface for key output



### `present_dnn/cipher/present_dnn.py`
**Full PRESENT-80 encryption pipeline** using PyTorch NN primitives:

- `present_encrypt(plaintext_int, key_int, key_bits=80)`: Main encryption function
- `present_encrypt_with_trace(plaintext_int, key_int, key_bits=80)`: Returns ciphertext + detailed trace

`present_encrypt`:
- Validates key (80-bit) and plaintext (64-bit) sizes
- Runs 31 rounds, each calling:
  - `XORNet` for addRoundKey
  - `SBoxLayer` applied to 16 nibbles
  - `PermutationLayer` for P-layer permutation
- Applies final whitening XOR with `K32`
- Returns ciphertext as integer

`present_encrypt_with_trace`:
- Returns `(ciphertext, trace_dict)` with per-round state snapshots
- Trace includes: round key, state after addRoundKey, after S-box, after P-layer
- Useful for debugging and verification

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

For one 64-bit plaintext block:

### Encryption Flow
1. **Bit Conversion**: Convert 64-bit plaintext integer to LSB-first bit vector using NumPy
2. **Key Schedule**: Compute all 32 round keys using key_schedule.py
   - Uses NumPy rotation + internal SBoxLayer (NN) + XORNet (NN) for updates
3. **Rounds (×31)**:
   - **AddRoundKey**: XOR with Ki using XORNet (ReLU-based 2-layer NN)
   - **SubBytes**: Apply SBoxLayer (corner-function 2-layer NN) to each 16 nibbles
   - **PermutationLayer**: Apply sparse P-matrix using 64×64 fixed sparse matrix
4. **Final Whitening**: XOR with K32 using XORNet
5. **Output**: Convert final bit vector back to 64-bit integer ciphertext

### Key Properties
- All operations are fixed (no learned/trainable parameters)
- All operations are deterministic (same input → same output)
- Uses PyTorch tensors (float32 in {0.0, 1.0}) with hardcoded weights
- Weights derived from PRESENT-80 specification, not learned

---

## 4) Running the Project

## 4.1 Install Dependencies

From repository root:

```bash
pip install -r present_dnn/requirements.txt
```

**Requirements**: PyTorch, NumPy

## 4.2 Run Official Verification Vectors

```bash
python -m present_dnn.utils.verify
```

Expected: all four vectors print `[PASS]`.

## 4.3 Run Tests

**Option 1: Custom test runner** (no pytest required):
```bash
python present_dnn/tests/run_tests.py
```

**Option 2: With pytest** (if available):
```bash
pytest present_dnn/tests/test_present_dnn.py -v
```

Current status: **All 6 tests passing** ✓

## 4.4 Custom Run: Hex Plaintext

```bash
python -m present_dnn.run_custom --key 00000000000000000000 --pt 0000000000000000
```

Output:
```
BLOCK 0
  PT: 0000000000000000
  CT: 5579c1387b228445
```

## 4.5 Custom Run: Hex Plaintext with Step Trace

```bash
python -m present_dnn.run_custom --key 00000000000000000000 --pt 0000000000000000 --trace
```

Trace includes `R01..R31` with per-round state snapshots (K, ARK, SB, PL), then final `R32` whitening.

## 4.6 Custom Run: Text Input

```bash
python -m present_dnn.run_custom --key 00000000000000000000 --text "Hi"
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

✅ **All correctness checks passing**:

### Official Appendix-I Test Vectors
| Plaintext | Key | Ciphertext |
|-----------|-----|------------|
| `0000000000000000` | `00000000000000000000` | `5579c1387b228445` ✓ |
| `0000000000000000` | `ffffffffffffffffffff` | `e72c46c0f5945049` ✓ |
| `ffffffffffffffff` | `00000000000000000000` | `a112ffc72f68417b` ✓ |
| `ffffffffffffffff` | `ffffffffffffffffffff` | `3333dcd3213210d2` ✓ |

### Component Tests
- **S-box**: Exactness verified for all 16 input nibbles via `SBoxLayer`
- **P-layer**: Permutation mapping verified for all 64 bit positions via `PermutationLayer`
- **Key Schedule**: Cross-checked against integer reference implementation for multiple keys
- **Round Keys**: Initial checks for all-zero master key (`K1=0x0000000000000000`, `K2=0xc000000000000000`)
- **Bit Conversions**: Roundtrip int↔bitvec verified for multiple values

---

## 6) Design Choices and Notes

### Numeric Representation
- **Tensor Type**: PyTorch `float32`
- **Bit Ordering**: **LSB-first arrays** throughout (bit 0 = least significant)
- **Binary Convention**: Bits represented as {0.0, 1.0} in float space

### Implementation Strategy
- **XOR**: ReLU-based exact identity: `ReLU(a-b) + ReLU(b-a)`
- **S-box**: Corner-function detection with 16 detection neurons + 4 output neurons
- **P-layer**: Sparse matrix multiplication (one 1.0 per column)
- **Key Schedule**: NumPy-based main logic with NN primitives for S-box and XOR operations
- **No Trainable Parameters**: All weights are fixed, deterministic derived from spec

### Key Schedule Notes
- Updates key register 31 times to produce round keys `K1..K32` 
- Final key `K32` is used for whitening after the 31 rounds
- Internal use of `SBoxLayer` and `XORNet` ensures NN consistency

---

## 7) Scope and Limitations

### Supported Features
- ✓ PRESENT-80 (80-bit key, 64-bit block)
- ✓ Full 31-round + final whitening encryption
- ✓ PyTorch tensor-based primitives with CPU/GPU support
- ✓ Detailed trace output for verification
- ✓ Official test vector validation

### Intentional Limitations
- **Key Size**: 80-bit only (PRESENT-128 not implemented)
- **Decryption**: Encryption only (decryption not implemented)
- **Modes**: Raw ECB-like per-block processing, no CBC/CTR/authenticated modes
- **CLI Text Mode**: Simple zero-padding, no proper mode of operation

### Security Note
For production use: combine with proper mode (CTR/GCM) + nonce + authentication. This implementation is for educational/research purposes with fixed operations.

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

## 10) Future Extensions (Not Yet Implemented)

### Algorithm Variants
- [ ] PRESENT-128 key schedule support
- [ ] Decryption path (inverse P-layer, inverse S-box, reverse key schedule)

### Enhancements
- [ ] GPU acceleration (move tensors to CUDA devices)
- [ ] nn.Module wrappers for easier PyTorch integration
- [ ] Batch processing utilities for multiple plaintexts
- [ ] Benchmark scripts: PyTorch NN vs NumPy vs C reference implementations

### Modes and Integration
- [ ] Mode-of-operation wrapper (CTR mode) with IV handling
- [ ] Authenticated encryption example
- [ ] Integration with PyTorch Lightning for training-based extensions
