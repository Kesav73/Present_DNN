# Key-Recovery Attack Steps and Fix Plan (PRESENT-DNN)

## Scope
This document explains, precisely, how the current epsilon-perturbation attack retrieves the secret key from the PRESENT-DNN design, and what to change to block it.

---

## 1) Exact attack assumptions
The attack succeeds when all of the following are true:

1. The attacker can query encryption repeatedly (chosen plaintexts).
2. Inputs are accepted as real-valued vectors (or can reach a real-valued internal path).
3. The first AddRoundKey behaves as a continuous XOR surrogate:
   - elementwise relation: `XOR(x, k) = |x - k|`
4. Final output preserves enough signal from first-round perturbations.

If inputs are strictly binary and enforced at boundary + each sensitive stage, this attack path is removed.

---

## 2) Step-by-step key retrieval

### Step 0: Build oracle access
Attacker needs one of:
- First-round oracle: `O1(x) = AddRoundKey(x, K1)` (strongest leakage), or
- Full encryption oracle: `E(x)` with real-valued query capability.

In your demo script, first-round oracle is used for clean recovery.

### Step 1: Choose a base plaintext vector
Pick a binary base vector:
- `x ∈ {0,1}^64`

### Step 2: For each target bit i, create paired perturbations
For each `i = 0..63`, define:
- `x_plus  = x + ε·e_i`
- `x_minus = x - ε·e_i`

Where:
- `ε > 0` is small (example: `0.01`)
- `e_i` is the unit vector with 1 at bit i

### Step 3: Query both inputs
Compute:
- `y_plus  = O1(x_plus)`
- `y_minus = O1(x_minus)`

(If using only final encryption, replace `O1` with `E` and use repeated trials/statistics.)

### Step 4: Measure output separation
Compute a distance score, e.g. L1 norm:
- `d_i = ||y_plus - y_minus||_1`

### Step 5: Infer K1 bit i
Using threshold `τ`:
- If `d_i <= τ`, infer `k1_i = x_i`
- If `d_i >  τ`, infer `k1_i = 1 - x_i`

Reason:
- For `|x_i - k_i|`, perturbation around matching/non-matching regions yields different local behavior, giving a separable signal.

### Step 6: Recover full first round key K1
After all 64 bit decisions:
- `K1 = (k1_63 ... k1_0)` (respect your implementation’s LSB-first indexing when converting)

### Step 7: Lift K1 to master key candidates
In your key schedule representation:
- `K1 = master_key[16:80]` (LSB-first array slice)
- Unknown bits are `master_key[0:16]`

So brute force only 16 bits:
- for `u in [0, 2^16 - 1]`:
  - `candidate_master = (K1 << 16) | u`

### Step 8: Verify candidate master key
Use 2–3 plaintext/ciphertext pairs:
- accept candidate if all pairs match oracle outputs

Expected work:
- Perturbation phase: about `2 * 64` queries (or multiplied by repeats)
- Search phase: at most `2^16` trials

---

## 3) Why this breaks intended 80-bit security
Original brute-force complexity target: `2^80`.

Attack reduces effort to:
- recover 64 key bits by differential leakage,
- brute-force only 16 bits (`2^16`).

So effective security collapses near the remaining 16-bit search.

---

## 4) Precise flaws and direct fixes

### Flaw A: Continuous-domain acceptance for cryptographic interface
- Problem: Real-valued inputs expose gradients/perturbation leakage.
- Fix: Enforce strict binary input domain at API boundary.
  - Reject anything not in `{0,1}` per bit (or exact integer block interface only).

### Flaw B: Continuous XOR surrogate exposed (`|x-k|`)
- Problem: Small perturbations around one coordinate leak key relation.
- Fix options:
  1. Replace security-critical Boolean ops with exact discrete bitwise ops.
  2. If NN form must remain, hard-quantize outputs after each primitive and reject non-binary inputs before entering primitive.

### Flaw C: Signal propagation through continuous layers
- Problem: Early leakage survives through rounds.
- Fix: Insert strict discretization/canonicalization barriers where cryptographic state is represented.

### Flaw D: Excess observability during development
- Problem: Trace/introspection functions can aid attacks if exposed.
- Fix: Keep trace utilities disabled in production and behind explicit debug flags.

### Flaw E: No explicit adversarial input policy
- Problem: Chosen-input real-value queries are not blocked.
- Fix: Add validation policy and tests that fail on non-binary / out-of-domain inputs.

---

## 5) Validation checklist after hardening

1. Non-binary plaintext queries are rejected.
2. First-round perturbation test no longer yields stable bit decisions.
3. `K1` cannot be reconstructed with threshold-based epsilon probing.
4. Brute-force stage is impossible without leaked `K1`.
5. Existing known-answer tests for PRESENT still pass for valid binary inputs.

---

## 6) Repro anchor in this repository
- Attack demo implementation: [present_dnn/attack_recover_key.py](present_dnn/attack_recover_key.py)
- Original attack explanation: [attacks.md](attacks.md)
- Cipher pipeline: [present_dnn/cipher/present_dnn.py](present_dnn/cipher/present_dnn.py)
- XOR primitive causing continuous behavior: [present_dnn/primitives/xor_layer_nn.py](present_dnn/primitives/xor_layer_nn.py)

---

## 7) Command to run the attack
From project root:

```bash
cd /home/netweb/vasu/Present_DNN
python3 -m present_dnn.attack_recover_key --key 0123456789abcdef0123 --eps 0.01 --threshold 1e-6 --pairs 3
```

Notes:
- Replace `--key` with your target 80-bit key (20 hex chars).
- Increase `--pairs` to tighten final key verification confidence.
