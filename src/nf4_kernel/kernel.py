import torch
import triton
import triton.language as tl

NF4_TABLE = [
    -1.0, -0.8525390625, -0.7021484375, -0.5693359375,
    -0.435791015625, -0.3039093017578125, -0.1741943359375, -0.0543212890625,
    0.0543212890625, 0.1741943359375, 0.3039093017578125, 0.435791015625,
    0.5693359375, 0.7021484375, 0.8525390625, 1.0,
]

NF4_TABLE_TENSOR = None


def _get_nf4_table(device):
    global NF4_TABLE_TENSOR
    if NF4_TABLE_TENSOR is None or NF4_TABLE_TENSOR.device != device:
        NF4_TABLE_TENSOR = torch.tensor(NF4_TABLE, dtype=torch.float32, device=device)
    return NF4_TABLE_TENSOR


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
def _nf4_kernel_bf16(packed_ptr, absmax_ptr, out_ptr, n_elem,
                     BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr):
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
    nibble = tl.where((offsets % 2) == 0, high, low).to(tl.int32)

    nf4_val = nf4_lookup_asm(nibble)

    absmax_offset = offsets // GROUP_SIZE
    absmax = tl.load(
        absmax_ptr + absmax_offset, mask=mask, other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = nf4_val * absmax
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)


@triton.jit
def _nf4_kernel_fp16(packed_ptr, absmax_ptr, out_ptr, n_elem,
                     BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr):
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
    nibble = tl.where((offsets % 2) == 0, high, low).to(tl.int32)

    nf4_val = nf4_lookup_asm(nibble)

    absmax_offset = offsets // GROUP_SIZE
    absmax = tl.load(
        absmax_ptr + absmax_offset, mask=mask, other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = nf4_val * absmax
    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)


def dequant_nf4(packed, absmax, group_size=64, dtype=torch.bfloat16):
    """Dequantize NF4-packed weights to bf16 or fp16.

    Args:
        packed: uint8 tensor with two 4-bit NF4 values per byte.
        absmax: block-wise absmax scaling factors (FP8 or float).
        group_size: number of elements sharing one absmax value (default 64).
        dtype: output dtype, must be torch.bfloat16 or torch.float16.

    Returns:
        Dequantized tensor of shape (packed.numel() * 2,) in the requested dtype.
    """
    n_elem = packed.numel() * 2
    expected_absmax = (n_elem + group_size - 1) // group_size
    if absmax.numel() != expected_absmax:
        raise ValueError(
            f"Expected absmax size {expected_absmax}, got {absmax.numel()}"
        )

    if dtype == torch.bfloat16:
        out = torch.empty(n_elem, dtype=torch.bfloat16, device=packed.device)
        kernel = _nf4_kernel_bf16
    elif dtype == torch.float16:
        out = torch.empty(n_elem, dtype=torch.float16, device=packed.device)
        kernel = _nf4_kernel_fp16
    else:
        raise ValueError(f"Unsupported output dtype: {dtype}. Use bfloat16 or float16.")

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elem, BLOCK_SIZE),)
    kernel[grid](packed, absmax, out, n_elem, BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size)
    return out


def quantize_nf4(tensor_bf16, block_size=64):
    """Quantize a bfloat16 tensor to NF4 format (for testing).

    Returns:
        packed: uint8 tensor with packed 4-bit NF4 indices.
        absmax_fp8: block-wise absmax as float8_e4m3fn.
        indices: raw NF4 indices (long tensor).
        absmax_f32: block-wise absmax as float32.
    """
    n_elem = tensor_bf16.numel()
    device = tensor_bf16.device
    nf4_t = _get_nf4_table(device)
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
    if all_indices.numel() % 2 != 0:
        all_indices = torch.cat([all_indices, torch.zeros(1, dtype=torch.long, device=device)])
    packed = ((all_indices[0::2] << 4) | all_indices[1::2]).to(torch.uint8)
    absmax_tensor = torch.stack(absmax_list)
    absmax_fp8 = absmax_tensor.to(torch.float8_e4m3fn)
    return packed, absmax_fp8, all_indices, absmax_tensor


def quantize_nf4_reference(tensor_bf16, indices, absmax_f32, group_size=64):
    """Reconstruct expected dequantized output from raw indices and absmax (for testing)."""
    nf4_t = _get_nf4_table(tensor_bf16.device)
    n_elem = tensor_bf16.numel()
    expected_nf4 = nf4_t[indices][:n_elem]
    absmax_expanded = absmax_f32.repeat_interleave(group_size)[:n_elem]
    return (expected_nf4 * absmax_expanded).to(tensor_bf16.dtype)
