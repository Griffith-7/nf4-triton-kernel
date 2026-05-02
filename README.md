# Unsloth AI Challenge #1: Optimized NF4 Triton Dequantization

## Overview
This repository contains the solution to the Unsloth AI Founding Engineer Challenge #1: "Convert NF4/BnB 4-bit to Triton".

The challenge is to convert the NF4 (NormalFloat 4) dequantization process from `bitsandbytes` (C++) into a highly optimized **Triton kernel**, achieving a speedup of at least 1.15x over the existing C++ implementation while maintaining full `torch.compile` compatibility.

## Results (Tesla T4)
- **Total Points:** 14/14 (Maximum Score)
- **Speedup:** 2.09x over bitsandbytes C++ baseline

| Feature | Status |
|---------|--------|
| Single Fused Kernel | ✅ (+3 points) |
| Inline PTX Assembly | ✅ (+3 points) |
| L2 Cache Eviction | ✅ (+1 point) |
| BF16/FP16 Output | ✅ (+1 point) |
| torch.compile Compatible | ✅ (+1 point) |
| >1.15x Speedup | ✅ 2.09x (+5 points) |

## Files
- `nf4_kernel_module.py`: Core Triton kernel with inline PTX assembly and L2 cache eviction
- `benchmark.py`: Benchmark script to measure performance against bitsandbytes
- `colab_notebook.py`: Complete Colab notebook code (cell-by-cell instructions)
- `unsloth_challenge_research.md`: Research notes on the challenge requirements

## Key Optimizations
1. **Inline PTX Assembly**: Uses `tl.inline_asm_elementwise` with predicated `mov.f32` instructions for the NF4 lookup table, avoiding branching overhead.
2. **L2 Cache Eviction**: Applies `eviction_policy="evict_first"` to all memory loads to prevent cache thrashing with large weight matrices.
3. **Fused Kernel**: Single pass for bit-unpacking, NF4 lookup, and block-wise absmax scaling.
4. **Dynamic Shapes**: Handles arbitrary tensor sizes without hardcoded grid/block dimensions.

## Benchmark
```
Triton kernel:  0.0515 ms
bitsandbytes:   0.1075 ms
SPEEDUP:        2.09x
```

## Requirements
- PyTorch 2.5+
- Triton 3.0+
- CUDA-capable GPU
