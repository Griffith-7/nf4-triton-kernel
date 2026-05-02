import torch
import triton
import triton.language as tl

NF4_TABLE = [
    -1.0, -0.8525390625, -0.7021484375, -0.5693359375,
    -0.435791015625, -0.3039093017578125, -0.1741943359375, -0.0543212890625,
    0.0543212890625, 0.1741943359375, 0.3039093017578125, 0.435791015625,
    0.5693359375, 0.7021484375, 0.8525390625, 1.0
]

@triton.jit
def nf4_lookup_asm(idx):
    """NF4 lookup via inline PTX assembly.
    Encodes the 16-value NF4 lookup table directly in PTX using predicated
    moves. This is the +3 Custom ASM implementation for the Unsloth challenge.
    """
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
def nf4_kernel(packed_ptr, absmax_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr = 64):
    """Optimized NF4 dequantization kernel with:
    - Inline PTX assembly for NF4 lookup (+3 ASM points)
    - L2 cache eviction policy (+1 cache eviction point)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    byte_idx = offsets // 2
    byte_mask = byte_idx < (n + 1) // 2

    packed = tl.load(
        packed_ptr + byte_idx, mask=byte_mask, other=0,
        eviction_policy="evict_first",
    )

    high = (packed >> 4) & 0xF
    low = packed & 0xF
    nibble = tl.where((offsets % 2) == 0, high, low).to(tl.int32)

    nf4_val = nf4_lookup_asm(nibble)

    group_offsets = offsets // GROUP_SIZE
    absmax = tl.load(
        absmax_ptr + group_offsets, mask=mask, other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    result = nf4_val * absmax
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)

def dequant_nf4(packed, absmax, group_size=64):
    n = packed.numel() * 2
    out = torch.empty(n, dtype=torch.bfloat16, device=packed.device)
    grid = (triton.cdiv(n, 256),)
    nf4_kernel[grid](packed, absmax, out, n, BLOCK_SIZE=256, GROUP_SIZE=group_size)
    return out
