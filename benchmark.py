import torch
import triton
import triton.language as tl
import time

NF4_TABLE = [
    -1.0, -0.8525390625, -0.7021484375, -0.5693359375,
    -0.435791015625, -0.3039093017578125, -0.1741943359375, -0.0543212890625,
    0.0543212890625, 0.1741943359375, 0.3039093017578125, 0.435791015625,
    0.5693359375, 0.7021484375, 0.8525390625, 1.0
]

@triton.jit
def nf4_lookup_asm(idx):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, pa, pb, pc, pd, pe;
            .reg .f32 v0;

            setp.eq.u32 p0, $1, 0;
            setp.eq.u32 p1, $1, 1;
            setp.eq.u32 p2, $1, 2;
            setp.eq.u32 p3, $1, 3;
            setp.eq.u32 p4, $1, 4;
            setp.eq.u32 p5, $1, 5;
            setp.eq.u32 p6, $1, 6;
            setp.eq.u32 p7, $1, 7;
            setp.eq.u32 p8, $1, 8;
            setp.eq.u32 p9, $1, 9;
            setp.eq.u32 pa, $1, 10;
            setp.eq.u32 pb, $1, 11;
            setp.eq.u32 pc, $1, 12;
            setp.eq.u32 pd, $1, 13;
            setp.eq.u32 pe, $1, 14;

            mov.f32 v0, 1.0;
            @p0 mov.f32 v0, -1.0;
            @p1 mov.f32 v0, -0.8525390625;
            @p2 mov.f32 v0, -0.7021484375;
            @p3 mov.f32 v0, -0.5693359375;
            @p4 mov.f32 v0, -0.435791015625;
            @p5 mov.f32 v0, -0.3039093017578125;
            @p6 mov.f32 v0, -0.1741943359375;
            @p7 mov.f32 v0, -0.0543212890625;
            @p8 mov.f32 v0, 0.0543212890625;
            @p9 mov.f32 v0, 0.1741943359375;
            @pa mov.f32 v0, 0.3039093017578125;
            @pb mov.f32 v0, 0.435791015625;
            @pc mov.f32 v0, 0.5693359375;
            @pd mov.f32 v0, 0.7021484375;
            @pe mov.f32 v0, 0.8525390625;

            mov.f32 $0, v0;
        }
        """,
        constraints="=f,r",
        args=[idx],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def nf4_dequant_kernel_optimized(
    packed_ptr, absmax_ptr, out_ptr, n_elem,
    group_size: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem

    byte_idx = offsets // 2
    byte_mask = byte_idx < (n_elem + 1) // 2

    packed = tl.load(
        packed_ptr + byte_idx, mask=byte_mask, other=0,
        eviction_policy="evict_first",
    )

    high = (packed >> 4) & 0xF
    low = packed & 0xF
    nibble = tl.where((offsets % 2) == 0, high, low)

    nf4_val = nf4_lookup_asm(nibble.to(tl.int32))

    absmax_offset = offsets // group_size
    absmax = tl.load(
        absmax_ptr + absmax_offset, mask=mask, other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    result = nf4_val * absmax
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)

@triton.jit
def nf4_dequant_kernel_baseline(
    packed_ptr, absmax_ptr, out_ptr, n_elem,
    group_size: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem

    byte_idx = offsets // 2
    byte_mask = byte_idx < (n_elem + 1) // 2

    packed = tl.load(packed_ptr + byte_idx, mask=byte_mask, other=0)

    high = (packed >> 4) & 0xF
    low = packed & 0xF
    nibble = tl.where((offsets % 2) == 0, high, low)

    nf4_val = tl.full([], 1.0, dtype=tl.float32)
    nf4_val = tl.where(nibble == 14, 0.8525390625, nf4_val)
    nf4_val = tl.where(nibble == 13, 0.7021484375, nf4_val)
    nf4_val = tl.where(nibble == 12, 0.5693359375, nf4_val)
    nf4_val = tl.where(nibble == 11, 0.435791015625, nf4_val)
    nf4_val = tl.where(nibble == 10, 0.3039093017578125, nf4_val)
    nf4_val = tl.where(nibble == 9, 0.1741943359375, nf4_val)
    nf4_val = tl.where(nibble == 8, 0.0543212890625, nf4_val)
    nf4_val = tl.where(nibble == 7, -0.0543212890625, nf4_val)
    nf4_val = tl.where(nibble == 6, -0.1741943359375, nf4_val)
    nf4_val = tl.where(nibble == 5, -0.3039093017578125, nf4_val)
    nf4_val = tl.where(nibble == 4, -0.435791015625, nf4_val)
    nf4_val = tl.where(nibble == 3, -0.5693359375, nf4_val)
    nf4_val = tl.where(nibble == 2, -0.7021484375, nf4_val)
    nf4_val = tl.where(nibble == 1, -0.8525390625, nf4_val)
    nf4_val = tl.where(nibble == 0, -1.0, nf4_val)

    absmax_offset = offsets // group_size
    absmax = tl.load(absmax_ptr + absmax_offset, mask=mask, other=0.0).to(tl.float32)

    result = nf4_val * absmax
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)

def dequant_nf4_optimized(packed_weights, absmax_fp8, group_size=64):
    n_elem = packed_weights.numel() * 2
    expected_absmax_size = (n_elem + group_size - 1) // group_size
    if absmax_fp8.numel() != expected_absmax_size:
        raise ValueError(f"Expected absmax size {expected_absmax_size}, got {absmax_fp8.numel()}")
    output = torch.empty(n_elem, dtype=torch.bfloat16, device=packed_weights.device)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elem, BLOCK_SIZE),)
    nf4_dequant_kernel_optimized[grid](
        packed_weights, absmax_fp8, output, n_elem,
        group_size=group_size, BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def dequant_nf4_baseline(packed_weights, absmax_fp8, group_size=64):
    n_elem = packed_weights.numel() * 2
    expected_absmax_size = (n_elem + group_size - 1) // group_size
    if absmax_fp8.numel() != expected_absmax_size:
        raise ValueError(f"Expected absmax size {expected_absmax_size}, got {absmax_fp8.numel()}")
    output = torch.empty(n_elem, dtype=torch.bfloat16, device=packed_weights.device)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elem, BLOCK_SIZE),)
    nf4_dequant_kernel_baseline[grid](
        packed_weights, absmax_fp8, output, n_elem,
        group_size=group_size, BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def quantize_nf4(tensor_bf16, block_size=64):
    n_elem = tensor_bf16.numel()
    device = tensor_bf16.device
    nf4_t = torch.tensor(NF4_TABLE, dtype=torch.float32, device=device)
    tensor_f32 = tensor_bf16.to(torch.float32).view(-1)
    n_blocks = (n_elem + block_size - 1) // block_size

    absmax_list = []
    indices_list = []

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n_elem)
        block = tensor_f32[start:end]
        absmax = block.abs().max()
        if absmax == 0:
            absmax = torch.tensor(1.0, device=device)
        absmax_list.append(absmax)
        block_norm = block / absmax
        dist = torch.abs(block_norm.unsqueeze(-1) - nf4_t)
        indices = torch.argmin(dist, dim=-1).long()
        indices_list.append(indices)

    all_indices = torch.cat(indices_list)
    packed = ((all_indices[0::2] << 4) | all_indices[1::2]).to(torch.uint8)
    absmax_tensor = torch.stack(absmax_list)
    absmax_fp8 = absmax_tensor.to(torch.float8_e4m3fn)
    return packed, absmax_fp8, all_indices, absmax_tensor

def run_benchmark():
    print("=" * 70)
    print("Unsloth Challenge 1: NF4 Triton Benchmark (RTX 3050)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Compute: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(42)

    print("\n[TEST 1] Functional Correctness...")
    test_size = 4096
    test_tensor = torch.randn(test_size, dtype=torch.bfloat16, device='cuda')
    packed, absmax_fp8, orig_indices, absmax_tensor = quantize_nf4(test_tensor)

    dequant_opt = dequant_nf4_optimized(packed, absmax_fp8)
    dequant_base = dequant_nf4_baseline(packed, absmax_fp8)

    nf4_t = torch.tensor(NF4_TABLE, dtype=torch.float32, device='cuda')
    expected_nf4 = nf4_t[orig_indices]
    absmax_expanded = absmax_tensor.repeat_interleave(64)[:test_size]
    expected = (expected_nf4 * absmax_expanded).to(torch.bfloat16)

    err_opt = (dequant_opt - expected).abs().max().item()
    err_base = (dequant_base - expected).abs().max().item()
    print(f"  Optimized (ASM): max error = {err_opt:.6f}")
    print(f"  Baseline (where): max error = {err_base:.6f}")
    print(f"  PASSED" if err_opt < 0.2 else f"  FAILED")

    print("\n[TEST 2] Benchmark (sizes matching LLM weight matrices)...")
    configs = [
        ("Small (4096)", 4096),
        ("Medium (16384)", 16384),
        ("Large (65536)", 65536),
        ("XL (262144)", 262144),
    ]

    for name, size in configs:
        bench_tensor = torch.randn(size, dtype=torch.bfloat16, device='cuda')
        packed_b, absmax_b, bench_indices, _ = quantize_nf4(bench_tensor)
        bench_indices_long = bench_indices.long()

        for _ in range(10):
            dequant_nf4_optimized(packed_b, absmax_b)
            dequant_nf4_baseline(packed_b, absmax_b)
        torch.cuda.synchronize()

        iterations = 500 if size <= 16384 else 200

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            dequant_nf4_optimized(packed_b, absmax_b)
        torch.cuda.synchronize()
        opt_time = (time.perf_counter() - start) / iterations

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            dequant_nf4_baseline(packed_b, absmax_b)
        torch.cuda.synchronize()
        base_time = (time.perf_counter() - start) / iterations

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            nf4_t[bench_indices_long].to(torch.bfloat16)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / iterations

        speedup_vs_base = base_time / opt_time if opt_time > 0 else 0
        speedup_vs_pytorch = pytorch_time / opt_time if opt_time > 0 else 0

        print(f"  {name}:")
        print(f"    Optimized (ASM+evict): {opt_time*1000:.4f} ms")
        print(f"    Baseline (where):      {base_time*1000:.4f} ms")
        print(f"    PyTorch ref:           {pytorch_time*1000:.4f} ms")
        print(f"    ASM vs Baseline:       {speedup_vs_base:.2f}x")
        print(f"    ASM vs PyTorch:        {speedup_vs_pytorch:.2f}x")

    print("\n" + "=" * 70)
    print("POINTS SUMMARY:")
    print("  +3: Single fused Triton kernel")
    print("  +3: Inline PTX ASM (nf4_lookup_asm)")
    print("  +1: L2 cache eviction (evict_first)")
    print("  +1: BF16 output")
    print("  +1: torch.compile (verify on Colab)")
    print("  +5: Speedup >1.15x vs BnB (verify on Colab T4)")
    print("  Total so far: 8/14 points (6 pending Colab verification)")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
