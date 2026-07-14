# NF4 Triton Kernel

<p align="center">
  <strong>Optimized NF4 (NormalFloat 4-bit) dequantization via Triton GPU kernels</strong>
</p>

<p align="center">
  <a href="https://github.com/Griffith-7/nf4-triton-kernel/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.13%2B-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/triton-3.6%2B-green.svg" alt="Triton">
  <a href="https://github.com/Griffith-7/nf4-triton-kernel/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
</p>

---

> **Unsloth AI Founding Engineer Challenge #1** — Convert NF4/BnB 4-bit dequantization from C++ to Triton.
> Achieves **1.27x–1.41x speedup** over `bitsandbytes` C++ across all tensor sizes.

```bash
pip install git+https://github.com/Griffith-7/nf4-triton-kernel.git
```

```python
from nf4_kernel import dequant_nf4

output = dequant_nf4(packed_weights, absmax_scales, dtype=torch.bfloat16)
```

---

## Why This Exists

LLMs use 4-bit NF4 quantization to fit in VRAM. At inference time, weights must be **dequantized** back to 16-bit. The existing `bitsandbytes` library does this in C++, but suffers from CPU dispatch overhead. This project implements the same operation as a **single fused Triton kernel** — eliminating the overhead and achieving consistent speedups.

## Benchmark Results

Tested against `bitsandbytes` v0.49.2 on **RTX 3050 Laptop GPU** (PyTorch 2.13.0, Triton 3.6.0):

| Tensor Size | Triton Kernel | bitsandbytes C++ | Speedup |
|-------------|---------------|-------------------|---------|
| 4,096 | 0.070 ms | 0.089 ms | **1.27x** |
| 16,384 | 0.067 ms | 0.088 ms | **1.32x** |
| 65,536 | 0.070 ms | 0.091 ms | **1.31x** |
| 262,144 | 0.066 ms | 0.093 ms | **1.41x** |

All sizes pass the **>1.15x** threshold. On Tesla T4, the original author reported **2.09x**.

## How It Works

A single fused Triton kernel performs three operations in one GPU pass:

```
uint8 packed bytes → bit unpack → NF4 table lookup (PTX ASM) → absmax scale → BF16/FP16 output
```

1. **Bit unpacking** — splits each `uint8` into two 4-bit nibbles via bit-shifting
2. **NF4 lookup** — maps each 4-bit index to a float using inline PTX assembly (16 values hardcoded in GPU registers)
3. **Absmax scaling** — multiplies by block-wise scaling factor and stores as BF16 or FP16

### Key Optimizations

| Optimization | Detail |
|--------------|--------|
| **Inline PTX Assembly** | NF4 lookup table lives in GPU registers via `tl.inline_asm_elementwise` — zero memory reads, no branch mispredictions |
| **L2 Cache Eviction** | `evict_first` for streaming packed weights, `evict_last` for shared absmax — prevents cache thrashing |
| **Single Fused Kernel** | Bit-unpack + lookup + scale in one pass — no intermediate memory round-trips |
| **Large Block Size** | 1024 elements per thread block for optimal GPU occupancy |
| **Dynamic Shapes** | Handles arbitrary tensor sizes without hardcoded grid/block dimensions |

## Feature Checklist

| Feature | Status | Points |
|---------|--------|--------|
| Single Fused Kernel | Passed | +3 |
| Inline PTX Assembly | Passed | +3 |
| L2 Cache Eviction | Passed | +1 |
| BF16/FP16 Output | Passed | +1 |
| torch.compile Compatible | Passed (inductor) | +1 |
| >1.15x Speedup vs BnB | Passed (1.27x – 1.41x) | +5 |
| **Total** | | **14/14** |

## Installation

```bash
# From GitHub
pip install git+https://github.com/Griffith-7/nf4-triton-kernel.git

# Editable (dev)
git clone https://github.com/Griffith-7/nf4-triton-kernel.git
cd nf4-triton-kernel
pip install -e ".[dev-all]"
```

### Requirements

- Python 3.10+
- PyTorch 2.13+
- Triton 3.6+
- CUDA 12.4+ GPU

## Project Structure

```
nf4-triton-kernel/
├── src/nf4_kernel/
│   ├── __init__.py          # Package exports, version
│   └── kernel.py            # Core Triton kernel + quantize/dequantize utils
├── tests/
│   └── test_nf4_kernel.py   # 21 unit tests (correctness, edge cases, validation)
├── benchmarks/
│   └── benchmark.py         # Performance benchmark vs bitsandbytes
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI (lint + test)
├── pyproject.toml           # PEP 621 project config
├── .pre-commit-config.yaml  # Pre-commit hooks (ruff, codespell, etc.)
├── Dockerfile               # Multi-stage build (CPU + CUDA)
├── colab_notebook.py        # Google Colab notebook (cell-by-cell)
└── README.md
```

## Usage

### Basic

```python
import torch
from nf4_kernel import dequant_nf4

# packed_weights: uint8 tensor (two 4-bit values per byte)
# absmax: block-wise scaling factors (FP8 or float32)
output = dequant_nf4(packed_weights, absmax, group_size=64, dtype=torch.bfloat16)
```

### FP16 Output

```python
output_fp16 = dequant_nf4(packed_weights, absmax, dtype=torch.float16)
```

### torch.compile

```python
@torch.compile
def compiled_dequant(pw, am):
    return dequant_nf4(pw, am)

output = compiled_dequant(packed_weights, absmax)
```

## Tests

```bash
python -m pytest tests/ -v
```

21 tests covering:
- Functional correctness (BF16, FP16)
- Edge cases (small/large/odd/non-aligned tensors, zeros, uniform, alternating)
- Quantize utility validation
- Input validation errors
- Multiple group sizes (32, 64, 128)

## Google Colab

```python
!git clone https://github.com/Griffith-7/nf4-triton-kernel.git
%cd nf4-triton-kernel
pip install -e .
exec(open("colab_notebook.py").read())
```

## License

MIT License — see [LICENSE](LICENSE).
