"""Unit tests for the NF4 Triton dequantization kernel.

Run with: python -m pytest test_nf4_kernel.py -v
Or directly: python test_nf4_kernel.py
"""

import torch
import pytest

from nf4_kernel import (
    dequant_nf4,
    quantize_nf4,
    quantize_nf4_reference,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


def _roundtrip_error(tensor_bf16, device, dtype=torch.bfloat16, group_size=64):
    """Quantize then dequantize and return max absolute error."""
    packed, absmax_fp8, indices, absmax_f32 = quantize_nf4(
        tensor_bf16, block_size=group_size
    )
    dequant = dequant_nf4(packed, absmax_fp8, group_size=group_size, dtype=dtype)
    expected = quantize_nf4_reference(tensor_bf16, indices, absmax_f32, group_size)
    return (dequant - expected.to(dtype)).abs().max().item()


class TestFunctionalCorrectness:
    def test_basic_dequantization_bf16(self, device):
        torch.manual_seed(42)
        t = torch.randn(4096, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device, dtype=torch.bfloat16)
        assert err < 0.2, f"BF16 max error {err} exceeds threshold"

    def test_basic_dequantization_fp16(self, device):
        torch.manual_seed(42)
        t = torch.randn(4096, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device, dtype=torch.float16)
        assert err < 0.2, f"FP16 max error {err} exceeds threshold"

    def test_output_is_bf16(self, device):
        torch.manual_seed(0)
        t = torch.randn(1024, dtype=torch.bfloat16, device=device)
        packed, absmax, _, _ = quantize_nf4(t)
        out = dequant_nf4(packed, absmax, dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16

    def test_output_is_fp16(self, device):
        torch.manual_seed(0)
        t = torch.randn(1024, dtype=torch.bfloat16, device=device)
        packed, absmax, _, _ = quantize_nf4(t)
        out = dequant_nf4(packed, absmax, dtype=torch.float16)
        assert out.dtype == torch.float16

    def test_output_shape(self, device):
        torch.manual_seed(0)
        t = torch.randn(1024, dtype=torch.bfloat16, device=device)
        packed, absmax, _, _ = quantize_nf4(t)
        out = dequant_nf4(packed, absmax)
        assert out.shape == (1024,)


class TestEdgeCases:
    def test_small_tensor_16_elements(self, device):
        """Minimum useful size: 16 elements = 8 bytes = 1 block."""
        t = torch.randn(16, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device)
        assert err < 0.2

    def test_odd_number_of_elements(self, device):
        """Odd count means last byte has only one nibble used."""
        n = 1023
        t = torch.randn(n, dtype=torch.bfloat16, device=device)
        packed, absmax_fp8, indices, absmax_f32 = quantize_nf4(t)
        dequant = dequant_nf4(packed, absmax_fp8, dtype=torch.bfloat16)
        expected = quantize_nf4_reference(t, indices, absmax_f32)
        err = (dequant[:n] - expected).abs().max().item()
        assert err < 0.2

    def test_exact_block_boundary(self, device):
        """Size exactly one block (64 elements)."""
        t = torch.randn(64, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device)
        assert err < 0.2

    def test_non_aligned_to_block(self, device):
        """Size not aligned to block size (100 elements)."""
        t = torch.randn(100, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device)
        assert err < 0.2

    def test_large_tensor(self, device):
        """Large tensor matching typical LLM weight sizes."""
        torch.manual_seed(42)
        t = torch.randn(262144, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device)
        assert err < 0.5

    def test_all_zeros(self, device):
        """Tensor of all zeros: NF4 maps 0 to nearest table value (-0.054)."""
        t = torch.zeros(256, dtype=torch.bfloat16, device=device)
        packed, absmax_fp8, indices, absmax_f32 = quantize_nf4(t)
        dequant = dequant_nf4(packed, absmax_fp8)
        expected = quantize_nf4_reference(t, indices, absmax_f32)
        err = (dequant - expected).abs().max().item()
        assert err < 0.01, f"Zero tensor roundtrip error {err} too large"

    def test_uniform_tensor(self, device):
        """Constant tensor should roundtrip accurately."""
        t = torch.full((256,), 0.5, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device)
        assert err < 0.2

    def test_alternating_sign(self, device):
        """Alternating positive/negative values."""
        vals = [1.0 if i % 2 == 0 else -1.0 for i in range(128)]
        t = torch.tensor(vals, dtype=torch.bfloat16, device=device)
        err = _roundtrip_error(t, device)
        assert err < 0.2


class TestQuantizeUtilities:
    def test_quantize_output_shapes(self, device):
        torch.manual_seed(0)
        t = torch.randn(256, dtype=torch.bfloat16, device=device)
        packed, absmax_fp8, indices, absmax_f32 = quantize_nf4(t)
        assert packed.shape == (128,), f"Expected packed shape (128,), got {packed.shape}"
        assert packed.dtype == torch.uint8
        assert absmax_fp8.dtype == torch.float8_e4m3fn
        assert indices.shape == (256,)
        assert absmax_f32.shape == (4,)

    def test_quantize_indices_in_valid_range(self, device):
        torch.manual_seed(0)
        t = torch.randn(512, dtype=torch.bfloat16, device=device)
        _, _, indices, _ = quantize_nf4(t)
        assert indices.min() >= 0
        assert indices.max() <= 15

    def test_packed_nibbles_valid(self, device):
        torch.manual_seed(0)
        t = torch.randn(512, dtype=torch.bfloat16, device=device)
        packed, _, _, _ = quantize_nf4(t)
        high_nibbles = (packed >> 4) & 0xF
        low_nibbles = packed & 0xF
        assert high_nibbles.max() <= 15
        assert low_nibbles.max() <= 15


class TestInputValidation:
    def test_wrong_absmax_size_raises(self, device):
        packed = torch.zeros(128, dtype=torch.uint8, device=device)
        wrong_absmax = torch.zeros(1, dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match="Expected absmax size"):
            dequant_nf4(packed, wrong_absmax)

    def test_unsupported_dtype_raises(self, device):
        packed, absmax, _, _ = quantize_nf4(
            torch.randn(256, dtype=torch.bfloat16, device=device)
        )
        with pytest.raises(ValueError, match="Unsupported output dtype"):
            dequant_nf4(packed, absmax, dtype=torch.float32)


class TestMultipleGroupSizes:
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_different_group_sizes(self, device, group_size):
        torch.manual_seed(42)
        n = 4096
        t = torch.randn(n, dtype=torch.bfloat16, device=device)
        packed, absmax_fp8, indices, absmax_f32 = quantize_nf4(
            t, block_size=group_size
        )
        dequant = dequant_nf4(packed, absmax_fp8, group_size=group_size)
        expected = quantize_nf4_reference(t, indices, absmax_f32, group_size)
        err = (dequant - expected).abs().max().item()
        assert err < 0.2, f"group_size={group_size}: error {err} exceeds threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
