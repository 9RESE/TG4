#!/usr/bin/env python3
"""
MIOpen Kernel Warmup Script

Pre-compiles MIOpen kernels to avoid runtime delays during training.
Run this script once after installing PyTorch or changing model configurations.

Usage:
    python -m ml.scripts.warmup_kernels
"""

import os
import sys
import time

# Set HSA override for RX 6700 XT before importing torch
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")

import torch
import torch.nn as nn


def warmup_lstm_kernels():
    """Warmup LSTM kernels for common configurations."""
    print("\n[LSTM Kernels]")

    device = 'cuda'
    batch_sizes = [32, 64, 128]
    seq_lengths = [60, 100, 200]
    hidden_sizes = [64, 128, 256]
    input_sizes = [10, 20, 32]

    total = len(batch_sizes) * len(seq_lengths) * len(hidden_sizes) * len(input_sizes)
    count = 0

    for batch in batch_sizes:
        for seq in seq_lengths:
            for hidden in hidden_sizes:
                for input_size in input_sizes:
                    lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden,
                        num_layers=2,
                        batch_first=True,
                        bidirectional=True
                    ).to(device)

                    x = torch.randn(batch, seq, input_size, device=device)
                    _ = lstm(x)
                    torch.cuda.synchronize()

                    count += 1
                    if count % 10 == 0:
                        print(f"  Progress: {count}/{total}")

                    del lstm, x
                    torch.cuda.empty_cache()

    print(f"  Compiled {total} LSTM kernel configurations")


def warmup_linear_kernels():
    """Warmup linear layer kernels."""
    print("\n[Linear Kernels]")

    device = 'cuda'
    batch_sizes = [32, 64, 128, 256]
    sizes = [64, 128, 256, 512, 1024]

    total = len(batch_sizes) * len(sizes) * len(sizes)
    count = 0

    for batch in batch_sizes:
        for in_size in sizes:
            for out_size in sizes:
                linear = nn.Linear(in_size, out_size).to(device)
                x = torch.randn(batch, in_size, device=device)
                _ = linear(x)
                torch.cuda.synchronize()

                count += 1
                del linear, x

    torch.cuda.empty_cache()
    print(f"  Compiled {total} linear kernel configurations")


def warmup_conv1d_kernels():
    """Warmup Conv1D kernels for temporal patterns."""
    print("\n[Conv1D Kernels]")

    device = 'cuda'
    batch_sizes = [32, 64, 128]
    seq_lengths = [60, 100, 200]
    channels = [16, 32, 64, 128]
    kernel_sizes = [3, 5, 7]

    total = len(batch_sizes) * len(seq_lengths) * len(channels) * len(kernel_sizes)
    count = 0

    for batch in batch_sizes:
        for seq in seq_lengths:
            for ch in channels:
                for ks in kernel_sizes:
                    conv = nn.Conv1d(ch, ch * 2, kernel_size=ks, padding=ks // 2).to(device)
                    x = torch.randn(batch, ch, seq, device=device)
                    _ = conv(x)
                    torch.cuda.synchronize()

                    count += 1
                    del conv, x

    torch.cuda.empty_cache()
    print(f"  Compiled {total} Conv1D kernel configurations")


def warmup_attention_kernels():
    """Warmup attention kernels for transformer models."""
    print("\n[Attention Kernels]")

    device = 'cuda'
    batch_sizes = [32, 64]
    seq_lengths = [60, 100]
    embed_dims = [64, 128, 256]
    num_heads = [4, 8]

    total = len(batch_sizes) * len(seq_lengths) * len(embed_dims) * len(num_heads)
    count = 0

    for batch in batch_sizes:
        for seq in seq_lengths:
            for embed in embed_dims:
                for heads in num_heads:
                    if embed % heads != 0:
                        continue

                    attn = nn.MultiheadAttention(
                        embed_dim=embed,
                        num_heads=heads,
                        batch_first=True
                    ).to(device)

                    x = torch.randn(batch, seq, embed, device=device)
                    _ = attn(x, x, x)
                    torch.cuda.synchronize()

                    count += 1
                    del attn, x

    torch.cuda.empty_cache()
    print(f"  Compiled {total} attention kernel configurations")


def warmup_activation_kernels():
    """Warmup activation function kernels."""
    print("\n[Activation Kernels]")

    device = 'cuda'
    activations = [nn.ReLU(), nn.GELU(), nn.SiLU(), nn.Tanh(), nn.Sigmoid()]
    sizes = [(64, 256), (128, 512), (256, 1024)]

    count = 0
    for act in activations:
        act = act.to(device)
        for size in sizes:
            x = torch.randn(*size, device=device)
            _ = act(x)
            torch.cuda.synchronize()
            count += 1
            del x

    torch.cuda.empty_cache()
    print(f"  Compiled {count} activation kernel configurations")


def warmup_batch_norm_kernels():
    """Warmup batch normalization kernels."""
    print("\n[BatchNorm Kernels]")

    device = 'cuda'
    sizes = [64, 128, 256, 512]
    batch_sizes = [32, 64, 128]

    count = 0
    for num_features in sizes:
        bn = nn.BatchNorm1d(num_features).to(device)
        bn.train()
        for batch in batch_sizes:
            x = torch.randn(batch, num_features, device=device)
            _ = bn(x)
            torch.cuda.synchronize()
            count += 1
            del x

    torch.cuda.empty_cache()
    print(f"  Compiled {count} BatchNorm kernel configurations")


def print_cache_info():
    """Print MIOpen cache information."""
    print("\n[Cache Information]")

    # Default MIOpen cache location
    default_cache = os.path.expanduser("~/.cache/miopen")

    print(f"  Default cache path: {default_cache}")
    print(f"  MIOPEN_USER_DB_PATH: {os.environ.get('MIOPEN_USER_DB_PATH', 'not set')}")
    print(f"  MIOPEN_CUSTOM_CACHE_DIR: {os.environ.get('MIOPEN_CUSTOM_CACHE_DIR', 'not set')}")

    if os.path.exists(default_cache):
        # Count files in cache
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(default_cache):
            for f in files:
                file_path = os.path.join(root, f)
                total_size += os.path.getsize(file_path)
                file_count += 1

        print(f"  Cache size: {total_size / 1e6:.1f} MB ({file_count} files)")
    else:
        print("  Cache directory does not exist yet")


def main():
    """Run kernel warmup."""
    print("=" * 60)
    print("   MIOpen Kernel Warmup")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n[ERROR] GPU not available. Cannot warmup kernels.")
        sys.exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

    start_time = time.time()

    # Warmup different kernel types
    warmup_linear_kernels()
    warmup_activation_kernels()
    warmup_batch_norm_kernels()
    warmup_lstm_kernels()
    warmup_conv1d_kernels()
    warmup_attention_kernels()

    # Print cache info
    print_cache_info()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Kernel warmup completed in {elapsed:.1f}s")
    print("       Subsequent training runs will be faster.")


if __name__ == "__main__":
    main()
