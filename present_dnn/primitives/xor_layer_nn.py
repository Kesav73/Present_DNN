"""XOR primitive as a PyTorch Neural Network module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class XORNet(nn.Module):
    """
    XOR implemented as a simple neural network.
    
    Architecture:
    - Input layer: 2 neurons (bits a and b)
    - Hidden layer: 2 neurons
      * Neuron 1: computes (a - b) with weights [1, -1]
      * Neuron 2: computes (b - a) with weights [-1, 1]
    - ReLU activation on hidden layer
    - Output layer: 1 neuron that sums both hidden outputs
      * Weights: [1, 1]
    
    Result: XOR(a, b) = ReLU(a - b) + ReLU(b - a)
    """
    
    def __init__(self, device=None, dtype=torch.float32):
        """
        Initialize XOR neural network.
        
        Args:
            device: Torch device (cpu or cuda)
            dtype: Data type for tensors
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Hidden layer: 2 inputs → 2 neurons (no bias)
        self.hidden = nn.Linear(2, 2, bias=False, device=device, dtype=dtype)
        
        # Output layer: 2 inputs → 1 neuron (no bias)
        self.output = nn.Linear(2, 1, bias=False, device=device, dtype=dtype)
        
        # Set fixed weights
        with torch.no_grad():
            # Hidden layer weights
            # Row 0: [1, -1]  → computes a - b
            # Row 1: [-1, 1]  → computes b - a
            self.hidden.weight = nn.Parameter(torch.tensor([
                [1.0, -1.0],
                [-1.0, 1.0]
            ], device=device, dtype=dtype))
            
            # Output layer weights: sum both hidden neurons
            # [1, 1] → output = ReLU(a-b) + ReLU(b-a) = XOR(a,b)
            self.output.weight = nn.Parameter(torch.tensor([
                [1.0, 1.0]
            ], device=device, dtype=dtype))
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute XOR for input tensors.
        
        Args:
            a: Binary tensor (values in {0.0, 1.0})
            b: Binary tensor (values in {0.0, 1.0}, same shape as a)
        
        Returns:
            XOR result tensor (same shape as input)
        """
        if a.shape != b.shape:
            raise ValueError("a and b must have identical shapes")
        
        # Stack inputs: shape becomes (..., 2)
        stacked = torch.stack([a, b], dim=-1)
        original_shape = stacked.shape[:-1]
        
        # Reshape to (batch_size, 2) for linear layer
        stacked_flat = stacked.reshape(-1, 2)
        
        # Forward pass through network
        h = self.hidden(stacked_flat)           # (batch, 2)
        h = F.relu(h)                            # ReLU activation
        output = self.output(h)                  # (batch, 1)
        
        # Reshape back to original shape and squeeze last dimension
        return output.reshape(original_shape)


class VectorXOR(nn.Module):
    """
    XOR for full bit vectors (e.g., 64-bit blocks).
    
    Applies XORNet element-wise to each bit pair in the vectors.
    """
    
    def __init__(self, vector_size: int, device=None, dtype=torch.float32):
        """
        Initialize vector XOR layer.
        
        Args:
            vector_size: Size of bit vectors (e.g., 64)
            device: Torch device (cpu or cuda)
            dtype: Data type for tensors
        """
        super().__init__()
        self.vector_size = vector_size
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Single XORNet module used for all bit positions
        self.xor_net = XORNet(device=device, dtype=dtype)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute XOR for entire bit vectors.
        
        Args:
            a: Binary tensor of size (..., vector_size)
            b: Binary tensor of size (..., vector_size)
        
        Returns:
            XOR result of same shape as input
        """
        return self.xor_net(a, b)





if __name__ == "__main__":
    print("Testing XOR Neural Network Implementations...\n")
    
    # Test 1: Simple 2-bit XOR
    print("1. Testing XORNet (2-bit input):")
    xor_net = XORNet()
    
    test_cases = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ]
    
    for a, b, expected in test_cases:
        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        result = xor_net(a_t, b_t).item()
        status = "✓" if abs(result - expected) < 0.1 else "✗"
        print(f"  {status} XOR({a}, {b}) = {result:.4f} (expected {expected})")
    
    # Test 2: Vector XOR (64-bit)
    print("\n2. Testing VectorXOR (64-bit vectors):")
    vec_xor = VectorXOR(vector_size=64)
    
    # Create test vectors
    a = torch.randint(0, 2, (64,), dtype=torch.float32)
    b = torch.randint(0, 2, (64,), dtype=torch.float32)
    result = vec_xor(a, b)
    
    print(f"  Input a shape: {a.shape}")
    print(f"  Input b shape: {b.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  First 10 bits of result: {result[:10].tolist()}")
    
    # Verify correctness by comparing to numpy
    print("\n3. Verifying against numpy XOR:")
    a_np = a.numpy()
    b_np = b.numpy()
    expected_xor = (a_np + b_np) % 2  # Numpy XOR equivalent
    result_np = result.detach().numpy()
    
    # Round to binary
    result_binary = (result_np + 0.5).astype(int) % 2
    matches = (result_binary == expected_xor).sum()
    print(f"  Matches: {matches}/{len(expected_xor)} bits")
    
    # Test 3: Batch processing
    print("\n4. Testing batch XOR (8 batches of 64-bit vectors):")
    batch_size = 8
    a_batch = torch.randint(0, 2, (batch_size, 64), dtype=torch.float32)
    b_batch = torch.randint(0, 2, (batch_size, 64), dtype=torch.float32)
    
    result_batch = vec_xor(a_batch, b_batch)
    print(f"  Batch input a shape: {a_batch.shape}")
    print(f"  Batch input b shape: {b_batch.shape}")
    print(f"  Batch output shape: {result_batch.shape}")
    
    print("\nAll tests completed!")

