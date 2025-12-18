#!/usr/bin/env python3
"""
GPU Test Script for AMD ROCm + PyTorch

Verifies that PyTorch can detect and use the AMD GPU for ML training.
Run this script to validate your GPU setup before training models.

Usage:
    python -m ml.scripts.test_gpu
"""

import os
import sys
import time

# Set HSA override for RX 6700 XT before importing torch
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")


def test_pytorch_gpu():
    """Test PyTorch GPU detection and basic operations."""
    import torch

    print("=" * 60)
    print("PyTorch GPU Test")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[ERROR] GPU not available!")
        print("Troubleshooting tips:")
        print("  1. Check that ROCm is installed: rocm-smi")
        print("  2. Verify user is in video/render groups: groups $USER")
        print("  3. Set HSA_OVERRIDE_GFX_VERSION=10.3.0 for RX 6700 XT")
        return False

    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Memory info
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1e9
    print(f"GPU memory: {total_memory:.1f} GB")
    print(f"GPU architecture: {props.name}")

    print("\n[PASS] PyTorch GPU detection successful!")
    return True


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    import torch

    print("\n" + "=" * 60)
    print("Tensor Operations Test")
    print("=" * 60)

    try:
        # Create tensors on GPU
        print("\nCreating tensors on GPU...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        # Matrix multiplication
        print("Testing matrix multiplication...")
        z = torch.mm(x, y)

        # Verify result
        print(f"Result shape: {z.shape}")
        print(f"Result device: {z.device}")
        print(f"Result dtype: {z.dtype}")

        # Clean up
        del x, y, z
        torch.cuda.empty_cache()

        print("\n[PASS] Tensor operations test successful!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Tensor operations failed: {e}")
        return False


def benchmark_matmul(size=4096, iterations=100):
    """Benchmark matrix multiplication performance."""
    import torch

    print("\n" + "=" * 60)
    print(f"Performance Benchmark (size={size}, iters={iterations})")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nRunning on: {device}")

    # Create tensors
    x = torch.randn(size, size, device=device, dtype=torch.float32)
    y = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warmup (important for MIOpen kernel compilation)
    print("Warming up (compiling kernels)...")
    for _ in range(10):
        _ = torch.mm(x, y)
    torch.cuda.synchronize()

    # Benchmark
    print("Running benchmark...")
    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Calculate TFLOPS
    flops = 2 * size ** 3 * iterations
    tflops = flops / elapsed / 1e12

    print(f"\nResults:")
    print(f"  Matrix size: {size}x{size}")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per iteration: {elapsed/iterations*1000:.2f}ms")
    print(f"  Performance: {tflops:.2f} TFLOPS (FP32)")

    # Expected performance for RX 6700 XT: ~8-10 TFLOPS FP32
    if tflops > 5.0:
        print("\n[PASS] Performance is within expected range!")
    else:
        print("\n[WARN] Performance is lower than expected. Check GPU utilization.")

    # Clean up
    del x, y
    torch.cuda.empty_cache()

    return tflops


def test_lstm_training():
    """Test LSTM model training on GPU."""
    import torch
    import torch.nn as nn

    print("\n" + "=" * 60)
    print("LSTM Training Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simple LSTM model
    class SimpleLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=10, hidden_size=64,
                               num_layers=2, batch_first=True)
            self.fc = nn.Linear(64, 3)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    try:
        # Create model and move to GPU
        print("\nCreating LSTM model on GPU...")
        model = SimpleLSTM().to(device)

        # Create dummy data
        batch_size = 64
        seq_len = 60
        features = 10
        x = torch.randn(batch_size, seq_len, features, device=device)
        y = torch.randint(0, 3, (batch_size,), device=device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        print("Running training iterations...")
        start = time.time()
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                print(f"  First epoch loss: {loss.item():.4f}")

        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  10 epochs completed in {elapsed:.2f}s")

        # Memory usage
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"  GPU memory allocated: {allocated:.1f} MB")
        print(f"  GPU memory reserved: {reserved:.1f} MB")

        print("\n[PASS] LSTM training test successful!")
        return True

    except Exception as e:
        print(f"\n[ERROR] LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_precision():
    """Test mixed precision (AMP) training."""
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler

    print("\n" + "=" * 60)
    print("Mixed Precision (AMP) Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[SKIP] GPU not available")
        return True

    try:
        device = 'cuda'

        # Simple model
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        ).to(device)

        x = torch.randn(64, 100, device=device)
        y = torch.randint(0, 3, (64,), device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()

        # Training with AMP
        print("\nRunning mixed precision training...")
        for i in range(5):
            optimizer.zero_grad()

            with autocast():
                output = model(x)
                loss = criterion(output, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"  Final loss: {loss.item():.4f}")
        print("\n[PASS] Mixed precision training successful!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Mixed precision failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)

    # Environment variables
    print("\nEnvironment Variables:")
    for var in ["HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ROCM_ARCH",
                "MIOPEN_FIND_MODE", "MIOPEN_USER_DB_PATH"]:
        value = os.environ.get(var, "not set")
        print(f"  {var}: {value}")

    # Python version
    print(f"\nPython version: {sys.version}")

    # Check for ROCm
    print("\nROCm Status:")
    try:
        import subprocess
        result = subprocess.run(["rocm-smi", "--showproductname"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
        else:
            print("  rocm-smi not available")
    except FileNotFoundError:
        print("  rocm-smi not found")


def main():
    """Run all GPU tests."""
    print("\n")
    print("=" * 60)
    print("   ML Trading System - GPU Test Suite")
    print("=" * 60)

    print_system_info()

    results = {}

    # Test 1: PyTorch GPU detection
    results['gpu_detection'] = test_pytorch_gpu()

    if not results['gpu_detection']:
        print("\n[ABORT] GPU detection failed. Cannot proceed with other tests.")
        sys.exit(1)

    # Test 2: Tensor operations
    results['tensor_ops'] = test_tensor_operations()

    # Test 3: Performance benchmark
    results['benchmark'] = benchmark_matmul() > 0

    # Test 4: LSTM training
    results['lstm_training'] = test_lstm_training()

    # Test 5: Mixed precision
    results['mixed_precision'] = test_mixed_precision()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All GPU tests passed! Ready for ML training.")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
