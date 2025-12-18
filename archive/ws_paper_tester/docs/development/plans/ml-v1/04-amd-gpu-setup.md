# AMD GPU Setup - ROCm & PyTorch Configuration

**Document Version**: 1.0
**Created**: 2025-12-16
**Status**: Setup Guide

---

## Overview

This document provides setup instructions for AMD GPU acceleration using ROCm and PyTorch on the TG4 trading system.

## Hardware Profile

| Component | Specification |
|-----------|---------------|
| GPU | AMD Radeon RX 6700 XT |
| VRAM | 12 GB GDDR6 |
| Architecture | RDNA 2 (gfx1031) |
| CPU | AMD Ryzen 9 7950X |
| RAM | 128 GB DDR5 |
| OS | Ubuntu Linux 6.8.0 |

## ROCm Compatibility

### Official Support Status

> **Note**: The RX 6700 XT is **not officially supported** by ROCm. However, it can work with additional configuration using the `HSA_OVERRIDE_GFX_VERSION` environment variable.

**Officially Supported GPUs** (ROCm 6.x):
- Radeon RX 7900 XTX/XT/GRE
- Radeon PRO W7900/W7800
- Instinct MI200/MI300 series

**Unofficially Working** (with workarounds):
- RX 6700 XT (gfx1031 → emulate gfx1030)
- RX 6800/6900 series (gfx1030)
- RX 5700 series (gfx1010)

## Installation Options

### Option 1: PIP Installation (Recommended)

AMD recommends PIP installation for development environments.

**Prerequisites**:
```bash
# Ensure Python 3.12 is available
python3 --version  # Should be 3.12.x

# Add user to required groups
sudo usermod -aG video,render $USER
# Log out and back in for group changes to take effect
```

**Install ROCm PyTorch**:
```bash
# Create virtual environment
python3 -m venv ~/.venvs/ml-trading
source ~/.venvs/ml-trading/bin/activate

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Or use AMD's tested wheels (more stable)
pip install torch==2.5.1+rocm6.2 \
    torchvision==0.20.1+rocm6.2 \
    torchaudio==2.5.1+rocm6.2 \
    --extra-index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/
```

**Environment Variables for RX 6700 XT**:
```bash
# Add to ~/.bashrc or activate script
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH="gfx1030"
```

### Option 2: Docker Installation (Most Reliable)

Docker provides a pre-tested, isolated environment.

```bash
# Pull ROCm PyTorch image
docker pull rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Run with GPU access
docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -v /home/rese/Documents/rese/trading-bots/grok-4_1:/workspace \
    rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0 \
    /bin/bash
```

**Docker Compose Integration**:
```yaml
# Add to docker-compose.yml
services:
  ml-training:
    image: rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    environment:
      - HSA_OVERRIDE_GFX_VERSION=10.3.0
      - DATABASE_URL=postgresql://user:pass@timescaledb:5432/trading
    volumes:
      - ./:/workspace
      - ml-models:/models
    working_dir: /workspace
    command: python train.py

volumes:
  ml-models:
```

### Option 3: Conda Installation

```bash
# Create conda environment
conda create -n ml-trading python=3.12
conda activate ml-trading

# Install PyTorch (ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Set environment variables
conda env config vars set HSA_OVERRIDE_GFX_VERSION=10.3.0
conda env config vars set PYTORCH_ROCM_ARCH=gfx1030
conda deactivate && conda activate ml-trading
```

## Verification

### Test GPU Detection

```python
# gpu_test.py
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test tensor operations
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.mm(x, y)
    print(f"Matrix multiplication test: PASSED")
    print(f"Result shape: {z.shape}")
```

**Expected Output**:
```
PyTorch version: 2.5.1+rocm6.2
ROCm available: True
Number of GPUs: 1
GPU Name: AMD Radeon RX 6700 XT
GPU Memory: 12.0 GB
Matrix multiplication test: PASSED
Result shape: torch.Size([1000, 1000])
```

### Benchmark Performance

```python
# benchmark.py
import torch
import time

def benchmark_matmul(size=4096, iterations=100):
    """Benchmark matrix multiplication"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # Warmup
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    for _ in range(10):
        _ = torch.mm(x, y)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    flops = 2 * size**3 * iterations
    tflops = flops / elapsed / 1e12
    print(f"Matrix size: {size}x{size}")
    print(f"Time: {elapsed:.2f}s for {iterations} iterations")
    print(f"Performance: {tflops:.2f} TFLOPS")

if __name__ == '__main__':
    benchmark_matmul()
```

**Expected Performance** (RX 6700 XT):
- ~8-10 TFLOPS FP32
- ~15-20 TFLOPS FP16 (with mixed precision)

## ML Framework Setup

### Install Additional Dependencies

```bash
# Activate environment
source ~/.venvs/ml-trading/bin/activate

# Install ML libraries
pip install \
    numpy<2 \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    stable-baselines3 \
    gymnasium \
    pandas-ta \
    optuna \
    pytorch-forecasting \
    tensorboard

# For data access
pip install \
    asyncpg \
    sqlalchemy \
    pyarrow \
    fastparquet
```

### Project Requirements Update

Update `requirements.txt`:
```
# ML Core
torch==2.5.1+rocm6.2
torchvision==0.20.1+rocm6.2
torchaudio==2.5.1+rocm6.2

# ML Libraries
xgboost>=2.0.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
optuna>=3.4.0
pytorch-forecasting>=1.0.0

# Technical Analysis
pandas-ta>=0.3.14b

# Data Processing
numpy<2
pandas>=2.0.0
pyarrow>=14.0.0

# Database
asyncpg>=0.29.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
```

## MIOpen Kernel Compilation

> **Important**: PyTorch uses MIOpen for ML primitives. Kernels are compiled at runtime, causing a warmup delay on first use.

### Pre-compile Kernels

```python
# warmup_kernels.py
"""Pre-compile MIOpen kernels to avoid runtime delays"""
import torch
import torch.nn as nn

def warmup_miopen():
    """Warmup common kernel configurations"""
    device = 'cuda'

    # Common sizes for trading ML
    batch_sizes = [32, 64, 128, 256]
    seq_lengths = [60, 100, 200]
    hidden_sizes = [64, 128, 256]

    print("Warming up MIOpen kernels...")

    # LSTM kernels
    for batch in batch_sizes:
        for seq in seq_lengths:
            for hidden in hidden_sizes:
                lstm = nn.LSTM(input_size=32, hidden_size=hidden,
                              num_layers=2, batch_first=True).to(device)
                x = torch.randn(batch, seq, 32, device=device)
                _ = lstm(x)
                torch.cuda.synchronize()

    # Linear kernels
    for batch in batch_sizes:
        for hidden in hidden_sizes:
            linear = nn.Linear(hidden, hidden).to(device)
            x = torch.randn(batch, hidden, device=device)
            _ = linear(x)
            torch.cuda.synchronize()

    # Conv1d kernels (for temporal patterns)
    for batch in batch_sizes:
        for seq in seq_lengths:
            conv = nn.Conv1d(32, 64, kernel_size=3, padding=1).to(device)
            x = torch.randn(batch, 32, seq, device=device)
            _ = conv(x)
            torch.cuda.synchronize()

    print("Kernel warmup complete!")

if __name__ == '__main__':
    warmup_miopen()
```

### Cache Location

MIOpen caches compiled kernels:
```bash
# Default cache location
~/.cache/miopen/

# Set custom cache directory
export MIOPEN_USER_DB_PATH=/path/to/cache
export MIOPEN_CUSTOM_CACHE_DIR=/path/to/cache
```

## Troubleshooting

### Common Issues

**1. GPU Not Detected**
```bash
# Check if GPU is visible
rocm-smi

# If not visible, check permissions
ls -la /dev/kfd /dev/dri/

# Add user to groups
sudo usermod -aG video,render $USER
# Log out and back in
```

**2. HSA Error on Unsupported GPU**
```bash
# Set override for RX 6700 XT
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Verify in Python
import os
print(os.environ.get('HSA_OVERRIDE_GFX_VERSION'))
```

**3. Out of Memory Errors**
```python
# Reduce batch size
batch_size = 64  # Instead of 256

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**4. Slow First Epoch**
```bash
# Pre-warm MIOpen kernels (see warmup script above)
python warmup_kernels.py

# Or use persistent kernel cache
export MIOPEN_FIND_MODE=3  # Use cached kernels only
```

### Performance Tips

1. **Use Mixed Precision (FP16)**
   - 2x faster training with half memory usage
   - Minimal accuracy impact for most models

2. **Optimize Batch Size**
   - Larger batches = better GPU utilization
   - RX 6700 XT (12GB): batch_size 64-256 typical

3. **Use DataLoader Workers**
   ```python
   DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True)
   ```

4. **Profile Your Code**
   ```python
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA]
   ) as prof:
       model(input)
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

## References

- [AMD ROCm Documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [ROCm GitHub - Beginner Setup Guide](https://github.com/RyanAhmed911/ml-amd-rocm-setup)
- [AMD Lab Notes - PyTorch Environment](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-pytorch-tensorflow-env-readme/)

---

**Next Document**: [ML Architecture](./05-ml-architecture.md)
