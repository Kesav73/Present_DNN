# 🔐 Key Extraction Attack on Neural Network Implementation of PRESENT Cipher

## 1. Introduction

This document explains how a secret key can be extracted from a neural network implementation of the PRESENT cipher.

In this setup, the cipher is implemented using fixed neural network layers (ReLU-based), where:
- XOR is implemented using ReLU functions
- S-box is implemented using corner functions
- Permutation is implemented using matrix operations

Although the implementation is **correct for binary inputs**, it operates over **real-valued inputs**, which introduces a critical vulnerability.

---

## 2. Cipher Configuration

- Block size: 64 bits  
- Key size: 80 bits  
- Number of rounds: 31  

The first step of encryption is:

state = plaintext ⊕ K₁

Where:
- K₁ is the first round key (64 bits)
- Derived from the 80-bit master key

---

## 3. Threat Model

The attacker has:

- Black-box access to the encryption function
- Ability to input real-valued inputs (not just 0 or 1)
- Access only to the final ciphertext

The attacker does NOT have:
- Access to intermediate states
- Access to round keys
- Access to model internals

---

## 4. Core Idea of the Attack

In neural network implementations, Boolean operations become continuous functions.

Example:

XOR(x, k) = |x - k|

This behaves correctly for binary inputs but leaks information for real-valued inputs.

---

## 5. Attack Strategy

The attack works by slightly perturbing inputs and observing how the output changes.

---

## 6. Step-by-Step Attack

### Step 1: Choose a Base Plaintext

P = [0, 1, 0, 1, ..., 1] (64-bit)

### Step 2: Target a Bit i

### Step 3: Create Perturbed Inputs

P⁺: x_i → x_i + ε  
P⁻: x_i → x_i - ε  

Example (ε = 0.01):
x_i = 0 → 0.01 and -0.01

### Step 4: Encrypt

C⁺ = Encrypt(P⁺)  
C⁻ = Encrypt(P⁻)

### Step 5: Compare Outputs

- If C⁺ ≈ C⁻ → k_i = x_i  
- If C⁺ ≠ C⁻ → k_i ≠ x_i  

---

## 7. Why It Works

- Difference introduced at first XOR
- Propagates through all rounds
- Neural network continuity preserves differences

---

## 8. Key Recovery

- Recover 64-bit round key K₁
- Remaining 16 bits brute-forced (2^16)

---

## 9. Complexity

- Bit recovery: O(64)
- Remaining brute force: 2^16

Security reduces from 2^80 → 2^16

---

## 10. Conclusion

This attack shows that neural network implementations of cryptographic primitives can be insecure, even if they are functionally correct for binary inputs.

Key takeaway:

Correctness on binary inputs ≠ Security in continuous domain