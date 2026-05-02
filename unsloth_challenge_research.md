# Unsloth AI: Founding ML Engineer Challenge Guide

## Overview

Unsloth AI, founded by Daniel Han, is known for making LLM training 2x faster and using 70% less memory. To recruit top-tier talent for their founding team, they launched a highly competitive "proof of work" hiring challenge. 

The challenge operates on a points-based system with very lucrative offers for those who can prove their deep technical expertise in PyTorch kernels, Triton, and LLM optimization.

**The Offers:**
- **$400K - $500K/yr + equity:** Founding Engineer (Requires **47 points**)
- **$250K - $300K/yr + equity:** ML Engineer (Requires **32 points**)

*No experience or PhD needed. The evaluation is purely based on the ability to solve the technical challenges.*

## The 5 Core Challenges

1. **Convert nf4 / BnB 4bit to Triton** (Highest-impact challenge)
2. **Make FSDP2 work with QLoRA**
3. **Remove graph breaks in torch.compile**
4. **Help solve Unsloth issues!** (Active OSS contributions)
5. **Memory Efficient Backprop**

---

## Deep Dive: Challenge #1 - Convert NF4 to Triton

The highest-impact challenge is converting the NF4 (NormalFloat 4) dequantization process from `bitsandbytes` (which is written in C++) into a highly optimized **Triton kernel**. This allows for deeper integration with PyTorch's `torch.compile` and eliminates C++ dispatch overhead, leading to massive speedups.

### The Objective
Implement a single, fused Triton kernel that performs NF4 weight dequantization (converting 4-bit weights to FP16 or BF16) faster and more efficiently than the existing `fast_dequantize` function in `bitsandbytes`.

### Strict Rules & Conditions

Before writing any code, you must adhere to these strict constraints to qualify for the points:

1. **Language Requirement (Triton Only):** The kernel **must** be written in OpenAI's **Triton** (a Python-based GPU language). Writing it in raw CUDA/C++ or Mojo disqualifies you from this specific challenge, as the goal is pure PyTorch integration.
2. **Zero Graph Breaks:** The kernel must be 100% compatible with `torch.compile`. If your implementation forces PyTorch to fall back to eager mode (a "graph break" jumping back to the CPU), you fail this requirement.
3. **Single Fused Kernel:** You cannot launch two separate operations. The dequantization of the scaling factors (`absmax`) and the 4-bit weights must happen in a single fused pass to minimize memory bandwidth usage.
4. **Data Type Strictness:**
   - **Input:** Weights arrive packed as `uint8` (two 4-bit values per byte). Scales arrive as `FP8`.
   - **Output:** The final dequantized tensor must be strictly `bfloat16` or `float16`.
5. **The Speed Penalty:** If your Triton kernel is slower than the existing C++ `bitsandbytes` implementation, you **lose 3 points**. You must achieve at least a **1.15x speedup** to gain the +5 speed points.
6. **Hardware Generality:** Your kernel must handle dynamic shapes. You cannot hardcode grid or block sizes that only work for one specific tensor size; it must calculate its pointers dynamically.

### Technical Roadmap

#### 1. The "Double Dequant" Math
Standard 4-bit quantization (NF4) stores both weights and scaling factors (`absmax`). To save memory, the scaling factors are also quantized (Double Quantization).
- **The Task:** Read the quantized `absmax` (FP8), dequantize it to FP32, and use it to dequantize the 4-bit weights into FP16 or BF16.
- **The Constraint:** Instead of launching separate kernels for `absmax` and weights, you must **fuse them into a single Triton kernel** to avoid memory round-trips.

#### 2. Bit Unpacking in Triton
NF4 packs two 4-bit weights into a single `uint8`. Your Triton kernel must load the `uint8` tensor and perform efficient bit-masking and shifting to extract the "nibbles":
```python
first_nibble = (packed_byte >> 4) & 0xF
second_nibble = packed_byte & 0xF
```

#### 3. Lookup Table (NF4 Map)
NF4 uses a specialized distribution lookup table of 16 values. You can either pass this table as a constant or hardcode it using `tl.inline_asm` for maximum performance.

#### 4. `torch.compile` Compatibility
The kernel must be fully compilable. You must avoid operations that cause a "Graph Break" (e.g., moving data back to the CPU or using non-Triton-friendly Python loops).

#### 5. Advanced Optimization: Cache Eviction
To squeeze out maximum points, implement an L2 cache management strategy to prevent the GPU from "thrashing" the cache while processing massive weight matrices.

### The Point Breakdown (Challenge #1 Scoring)

The scoring is extremely strict. If your kernel is slower than `bitsandbytes`, you will *lose* points. To gain maximum points, you must beat the baseline speed.

| Task Component | Points |
| :--- | :--- |
| **Single Triton Kernel** | +3 |
| **Speedup (>= 1.15x faster than BnB)** | +5 (cumulative) |
| **Works in `torch.compile`** | +1 |
| **Custom ASM (Inline Assembly)** | +3 |
| **Cache Eviction Strategy** | +1 |
| **BF16/FP16 Support** | +1 |

---

## Understanding the 47 Points System

The **47 points** requirement for the Founding Engineer role is an accumulative score across all the challenges. You do not strictly need to complete all 5 challenges to hit 47 points, but you must achieve near-perfect scores on the ones you do tackle.

For example, Challenge 1 (Triton NF4) is worth up to **14 points** if perfectly executed (speedup, cache eviction, inline ASM, `torch.compile` compatibility). To get the remaining 33 points, you would need to tackle other high-value challenges like FSDP2/QLoRA integration or actively solve major bugs on their GitHub repository (Challenge #4). 

**Has anyone achieved the 47 points?**
Yes, since this is an ongoing hiring process, a few elite engineers have landed the Founding Engineer roles. However, because Unsloth continues to hire and post bug bounties, the "Founding Team" is actively expanding. The baseline times keep getting faster, meaning the 47 points are harder to achieve today than they were when the challenge first launched.

## Existing Community Solutions vs. Our Superior Strategy

Several engineers, such as **Sambhav Dixit (Indosambhav)** and **RameshBabuAsh**, have published public solutions to Challenge 1. 

**How they did it:**
- They successfully fused the "double dequantization" of the FP8 `absmax` and the 4-bit weights into a single Triton kernel.
- They used basic bitwise shifts (`(packed_byte >> 4) & 0xF`) to unpack the `uint8` tensors.
- They achieved `torch.compile` compatibility by removing C++ dispatch overhead.

**How we can do better (and hit the maximum 14 points for Challenge 1):**
While existing solutions are good, we must beat Unsloth's *current* internal baseline to get the speedup points. We will out-engineer the public solutions by focusing on:
1. **Custom PTX Assembly (`tl.inline_asm`):** Instead of passing the NF4 16-value lookup table as a tensor, we will hardcode it directly into GPU registers using inline assembly. This guarantees the **+3 points** for ASM and massively reduces memory latency.
2. **L2 Cache Eviction Control:** We will explicitly use Triton's `evict_first` or `evict_last` hints to manage the GPU's L2 cache. This prevents massive weight matrices from "thrashing" the cache, earning an extra **+1 point** and boosting overall LLM forward pass speeds.
3. **Perfect Memory Coalescing:** We will ensure all `tl.load` operations are 128-byte aligned and vectorized specifically for Hopper/Blackwell architectures.

## Strategy for Hitting 47 Points

To hit the 47-point threshold, you must combine elite kernel work with real-world impact:

1. **Benchmark First:** Use the official Unsloth Challenge Colab to measure the current `bitsandbytes` speed on a T4 or A100. This is your baseline to beat. 
2. **Nail the Triton Kernel:** Achieve the >1.15x speedup, inline assembly, and cache eviction for maximum points on Challenge 1.
3. **Solve GitHub Issues:** Combine your high-level kernel work with active Open Source contributions (Challenge #4). Fixing 2-3 "Good First Issues" or complex bugs on the Unsloth GitHub repo is the most practical way to accumulate the remaining points required for the Founding Engineer tier.

## Application Tips

- **Proof of Work is King:** Link a Colab notebook or GitHub repository proving your solution. Do not just talk about ML concepts; show raw kernel code, memory management, and math.
- **Focus on Low-Level Mechanics:** Emphasize your understanding of memory bandwidth, Triton (`tl.load`, masking), instruction-level parallelism, and PyTorch internals.
- **Stand Out:** Even if others have solved the challenge publicly, your goal is to be the *best*, not just the first. If your implementation handles memory better or integrates more cleanly than community solutions, you will get noticed.

## Important Links & Resources
- **Official Submission Colab:** [Unsloth Challenge Colab](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing)
- **Unsloth GitHub Repo:** [unslothai/unsloth](https://github.com/unslothai/unsloth)
- **Job Listing:** [Unsloth AI on Work at a Startup](https://www.workatastartup.com/jobs/73175)
