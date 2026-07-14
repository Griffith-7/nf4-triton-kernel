"""NF4 Triton Kernel — Optimized NF4 dequantization via Triton GPU kernels."""

__version__ = "1.0.0"

from .kernel import (
    NF4_TABLE,
    dequant_nf4,
    nf4_lookup_asm,
    quantize_nf4,
    quantize_nf4_reference,
)

__all__ = [
    "NF4_TABLE",
    "dequant_nf4",
    "nf4_lookup_asm",
    "quantize_nf4",
    "quantize_nf4_reference",
]
