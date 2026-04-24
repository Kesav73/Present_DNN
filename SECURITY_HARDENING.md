# Security Hardening Guide: Protecting PRESENT-DNN from Key Recovery Attacks

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Attack Summary](#attack-summary)
3. [Security Fixes](#security-fixes)
4. [Implementation Guide](#implementation-guide)
5. [Validation Checklist](#validation-checklist)
6. [Testing Strategy](#testing-strategy)

---

## Problem Overview

The PRESENT-DNN implementation is **cryptographically correct** for binary inputs but contains a critical vulnerability: it accepts **real-valued (continuous) inputs**, which exposes the encryption key through differential perturbation attacks.

### Impact
- **Theoretical Security**: 2^80 possible keys (exhaustive search)
- **Actual Security with Vulnerability**: ~2^16 (after 65K operations)
- **Security Loss**: 10^24× reduction

### Root Cause
The implementation uses neural network operations (ReLU-based XOR, corner-function S-boxes) that are **continuous and differentiable**. While these work correctly for binary inputs {0,1}, accepting real-valued inputs creates exploitable information leakage.

---

## Attack Summary

### How the Attack Works

**Stage 1: Differential Perturbation (65 queries)**
1. Attacker sends a base plaintext **P** with real-valued entries
2. For each of the 64 bits, creates **P⁺** (bit + ε) and **P⁻** (bit - ε)
3. Observes the magnitude of output differences
4. Uses threshold decision: small difference → key bit matches plaintext bit; large difference → key bit is opposite
5. Recovers all 64 bits of K₁ (first round key)

**Stage 2: Brute-Force (2^16 = 65K trials)**
1. Key schedule reveals that only 16 bits of the 80-bit master key are unknown
2. Tries all 2^16 candidates using 2-3 plaintext/ciphertext pairs for verification
3. Finds the correct key

### Why It Succeeds

The XOR operation as a continuous function:
$$\text{XOR}(x, k) = |x - k|$$

This function is **differentiable** and **continuous**:
- When **x_i ≈ k_i**: perturbations have small effect (low derivative)
- When **x_i ≠ k_i**: perturbations have large effect (high derivative)

This local behavior leaks the key bit relationship.

---

## Security Fixes

### Fix #1: Strict Binary Input Validation (CRITICAL)

**What it does**: Rejects any input that is not strictly in {0.0, 1.0}

**Why it's needed**:
- Prevents attackers from using continuous/real-valued inputs
- Eliminates the entire attack surface for perturbation-based key recovery
- Enforces the intended cryptographic model (Boolean algebra)

**Implementation**:
```python
def _validate_binary_input(value: torch.Tensor, name: str = "input") -> None:
    """Ensure all values are strictly binary {0.0, 1.0}."""
    if not torch.all((value == 0.0) | (value == 1.0)):
        raise ValueError(
            f"{name} must be strictly binary (all values in {{0.0, 1.0}}). "
            f"Got range [{value.min().item():.6f}, {value.max().item():.6f}]"
        )
```

**Where to add**:
- At the entry point of `present_encrypt()` after plaintext conversion
- Before any cryptographic operation begins

**Impact**: ✅ Blocks perturbation attacks entirely

---

### Fix #2: Replace Continuous XOR with Discrete Bitwise XOR (CRITICAL)

**What it does**: Replaces ReLU-based XOR approximation with PyTorch's exact logical XOR

**Why it's needed**:
- The ReLU-based XOR (`|x - k|`) is inherently continuous
- Even with binary inputs, the internal representations can be manipulated
- Discrete XOR has no gradient/derivative—no information leakage
- Mathematical correctness: true Boolean XOR operation

**Current vulnerable code**:
```python
state = xor_nn(state, round_key)  # Uses ReLU: XOR(a,b) = ReLU(a-b) + ReLU(b-a)
```

**Hardened code**:
```python
state = torch.logical_xor(state.bool(), round_key.bool()).float()
```

**Why this works**:
- `torch.logical_xor()` operates on Boolean values, not real numbers
- No gradient information is exposed
- Result is guaranteed binary {0.0, 1.0}
- Cryptographically equivalent to standard XOR

**Impact**: ✅ Eliminates continuous XOR leakage path

---

### Fix #3: Hard Quantization After Each Primitive (IMPORTANT)

**What it does**: Rounds all intermediate values to {0.0, 1.0} after S-box and permutation layers

**Why it's needed**:
- Even if individual operations are discrete, accumulated floating-point rounding errors can create continuous artifacts
- S-box and permutation implementations may have small numerical deviations
- Quantization ensures state remains strictly binary throughout all rounds
- Prevents information leakage through numerical precision

**Implementation**:
```python
def _quantize_to_binary(tensor: torch.Tensor) -> torch.Tensor:
    """Round all values to nearest binary {0.0, 1.0}."""
    quantized = torch.round(tensor)
    # Verify result is valid
    if not torch.all((quantized == 0.0) | (quantized == 1.0)):
        raise RuntimeError("Quantization produced non-binary values")
    return quantized
```

**Where to add** (in main round loop):
```python
for round_idx in range(31):
    # AddRoundKey
    state = torch.logical_xor(state.bool(), round_key.bool()).float()
    state = _quantize_to_binary(state)  # ← ADD HERE
    
    # S-box Layer
    post_sbox = torch.zeros(64, dtype=torch.float32)
    for i in range(16):
        nibble = state[4*i:4*i+4].unsqueeze(0)
        post_sbox[4*i:4*i+4] = sbox_nn(nibble).squeeze(0)
    state = _quantize_to_binary(post_sbox)  # ← ADD HERE
    
    # Permutation Layer
    state = player_nn(state)
    state = _quantize_to_binary(state)  # ← ADD HERE
```

**Impact**: ✅ Prevents accumulation of continuous numerical artifacts

---

### Fix #4: Disable Trace Functions in Production (RECOMMENDED)

**What it does**: Restricts access to intermediate states only in explicit debug mode

**Why it's needed**:
- `present_encrypt_with_trace()` exposes round-by-round states
- An attacker with trace access can perform differential attacks more efficiently
- Production deployments should not expose internal state
- Debug features should be explicitly opted-in

**Implementation**:
```python
DEBUG_MODE = False  # Set to True only during development

def present_encrypt_with_trace(plaintext_int: int, key_int: int, 
                               debug_mode: bool = None) -> tuple:
    """Encrypt with optional trace output."""
    if debug_mode is None:
        debug_mode = DEBUG_MODE
    
    if not debug_mode:
        raise SecurityWarning(
            "Trace output is disabled in production mode. "
            "Set DEBUG_MODE=True only during development."
        )
    # ... rest of implementation ...
```

**Alternative approach** (stricter):
```python
def present_encrypt_with_trace(plaintext_int: int, key_int: int) -> tuple:
    """Trace only available in test environment."""
    import os
    if os.environ.get('PRESENT_DEBUG') != 'true':
        raise SecurityWarning("Trace disabled in production. Set PRESENT_DEBUG=true to enable.")
    # ... trace implementation ...
```

**Impact**: ✅ Reduces information available to attackers

---

### Fix #5: Add Input Validation Tests (IMPORTANT)

**What it does**: Comprehensive test suite ensuring non-binary inputs are rejected

**Why it's needed**:
- Validates that the binary input enforcement is working
- Catches regressions if code is modified later
- Ensures edge cases are handled (NaN, Inf, values > 1, negative values)
- Documents the security boundary

**Implementation**:
```python
def test_reject_non_binary_plaintext():
    """Verify that non-binary plaintext inputs are rejected."""
    key = 0x0123456789abcdef0123
    
    # Test 1: Plaintext with value > 1
    with pytest.raises(ValueError, match="strictly binary"):
        plaintext_float = torch.tensor([2.0, 1.0, 0.0, 1.0], ...)
        validate_binary_input(plaintext_float, "plaintext")
    
    # Test 2: Plaintext with fractional values
    with pytest.raises(ValueError, match="strictly binary"):
        plaintext_float = torch.tensor([0.5, 1.0, 0.0, 1.0], ...)
        validate_binary_input(plaintext_float, "plaintext")
    
    # Test 3: Negative values
    with pytest.raises(ValueError, match="strictly binary"):
        plaintext_float = torch.tensor([-0.1, 1.0, 0.0, 1.0], ...)
        validate_binary_input(plaintext_float, "plaintext")

def test_reject_epsilon_perturbations():
    """Ensure small epsilon perturbations are rejected."""
    key = 0x0123456789abcdef0123
    base_plaintext = torch.tensor([0.0, 1.0, 0.0, 1.0, ...], ...)
    
    # Try adding epsilon
    perturbed = base_plaintext.clone()
    perturbed[0] += 0.01  # Add epsilon
    
    with pytest.raises(ValueError, match="strictly binary"):
        present_encrypt(perturbed, key)

def test_valid_binary_inputs_accepted():
    """Verify legitimate binary inputs still work."""
    key = 0x0123456789abcdef0123
    plaintext_int = 0x0123456789abcdef
    
    # Should not raise
    ciphertext = present_encrypt(plaintext_int, key)
    assert isinstance(ciphertext, int)
    assert 0 <= ciphertext < (1 << 64)
```

**Impact**: ✅ Defines and enforces security boundary

---

## Implementation Guide

### Step 1: Update `present_dnn/cipher/present_dnn.py`

Add validation function at the top:
```python
def _validate_binary_input(value: torch.Tensor, name: str = "input") -> None:
    """Ensure all values are strictly binary {0.0, 1.0}."""
    if not torch.all((value == 0.0) | (value == 1.0)):
        min_val = value.min().item()
        max_val = value.max().item()
        raise ValueError(
            f"{name} must be strictly binary (all values in {{0.0, 1.0}}). "
            f"Got range [{min_val:.6f}, {max_val:.6f}]"
        )

def _quantize_to_binary(tensor: torch.Tensor) -> torch.Tensor:
    """Round all values to nearest binary {0.0, 1.0}."""
    quantized = torch.round(tensor)
    if not torch.all((quantized == 0.0) | (quantized == 1.0)):
        raise RuntimeError(
            "Quantization failed: non-binary values present after rounding"
        )
    return quantized
```

Update `present_encrypt()`:
- Add `_validate_binary_input(state, "plaintext")` after converting plaintext
- Replace `xor_nn()` calls with `torch.logical_xor()`
- Add `_quantize_to_binary()` after each layer

### Step 2: Update Tests

Add the validation tests to `present_dnn/tests/test_present_dnn.py`:
```python
import pytest

def test_reject_non_binary_plaintext():
    """Verify non-binary inputs are rejected."""
    # ... implementation from Fix #5 ...

def test_reject_epsilon_perturbations():
    """Ensure epsilon perturbations are rejected."""
    # ... implementation from Fix #5 ...

def test_valid_binary_inputs_accepted():
    """Verify legitimate binary inputs still work."""
    # ... implementation from Fix #5 ...
```

### Step 3: Update Documentation

Add to `run_custom.py` or create `SECURITY_NOTES.md`:
```markdown
## Security Notes

- Input must be strictly binary {0, 1} at the bit level
- Real-valued inputs are rejected with ValueError
- Trace output is only available in debug mode
- All intermediate states are quantized to prevent numerical leakage
```

---

## Validation Checklist

After implementing all fixes, verify:

- [ ] **Non-binary rejection**: Any input with value ∉ {0.0, 1.0} raises `ValueError`
  ```bash
  python -c "from present_dnn.cipher.present_dnn import present_encrypt; \
    import torch; present_encrypt(torch.tensor([0.5, 1.0, ...]), 0x1234567890abcdef0123)"
  # Should raise: ValueError: ...strictly binary...
  ```

- [ ] **Epsilon perturbations fail**: Continuous attack no longer works
  ```bash
  python -m present_dnn.attack_recover_key --key 0123456789abcdef0123 --eps 0.01
  # Should raise: ValueError: ...strictly binary...
  ```

- [ ] **Test vectors still pass**: Encryption correctness unchanged
  ```bash
  python -m present_dnn.tests.run_tests
  # All 4 PRESENT test vectors should PASS
  ```

- [ ] **Quantization preserves correctness**: Binary inputs produce identical output
  ```python
  plaintext, key = 0x0123456789abcdef, 0x0123456789abcdef0123
  ct1 = present_encrypt(plaintext, key)  # Before adding quantization
  ct2 = present_encrypt(plaintext, key)  # After adding quantization
  assert ct1 == ct2  # Should be identical
  ```

- [ ] **Trace disabled in production**: Accessing trace without debug mode fails
  ```bash
  python -c "from present_dnn.cipher.present_dnn import present_encrypt_with_trace; \
    present_encrypt_with_trace(0x1234, 0x1234567890abcdef0123, debug_mode=False)"
  # Should raise: SecurityWarning
  ```

---

## Testing Strategy

### Unit Tests
```bash
cd /home/netweb/vasu/Present_DNN
python -m pytest present_dnn/tests/test_present_dnn.py -v
```

### Known-Answer Tests
```bash
python -m present_dnn.tests.run_tests
```

### Attack Verification
```bash
# Before hardening: Should succeed in recovering key
python -m present_dnn.attack_recover_key --key 0123456789abcdef0123 --eps 0.01

# After hardening: Should fail with "strictly binary" error
python -m present_dnn.attack_recover_key --key 0123456789abcdef0123 --eps 0.01
```

### Regression Testing
After implementing fixes, ensure:
1. All official test vectors still pass
2. Binary encryption is unchanged
3. Invalid inputs are consistently rejected
4. Error messages are clear and actionable

---

## Summary

| Fix | Severity | Lines Changed | Attack Coverage | Dependencies |
|-----|----------|---------------|-----------------|--------------|
| Input Validation | CRITICAL | ~20 | Blocks perturbation entirely | None |
| Discrete XOR | CRITICAL | ~3 | Eliminates gradient leakage | PyTorch |
| Quantization | IMPORTANT | ~15 | Prevents numerical artifacts | None |
| Trace Gating | RECOMMENDED | ~10 | Reduces observability | os module |
| Test Suite | IMPORTANT | ~40 | Validates all above | pytest |

**Total effort**: ~100 lines of code across 3 files

**Security gain**: From 2^16 effective brute-force → 2^80 (full security restored)

---

## References

- **Attack Details**: [attacks.md](attacks.md)
- **Step-by-Step Attack**: [KEY_RECOVERY_ATTACK_STEPS.md](KEY_RECOVERY_ATTACK_STEPS.md)
- **Demo Implementation**: [present_dnn/attack_recover_key.py](present_dnn/attack_recover_key.py)
- **Original Cipher**: [present_dnn/cipher/present_dnn.py](present_dnn/cipher/present_dnn.py)
- **XOR Primitive**: [present_dnn/primitives/xor_layer_nn.py](present_dnn/primitives/xor_layer_nn.py)
