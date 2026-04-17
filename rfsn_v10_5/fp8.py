"""Software FP8 helpers for runtimes without native float8 support.

The current MLX runtime in this workspace does not expose a native
``float8_e4m3`` dtype, but we still want the cache to store one byte per
value when ``cache_dtype='fp8_e4m3'``. This module provides a software
fallback that packs tensors into ``uint8`` using a finite-only E4M3-style
layout and unpacks them back to a normal MLX floating-point dtype.

The implementation is fully vectorized and stays inside MLX, so the
cache encode/decode path does not have to round-trip through NumPy or
other host-side tensor transforms.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


FP8_E4M3_EXPONENT_BIAS = 7
FP8_E4M3_MIN_NORMAL = 2.0 ** -6
FP8_E4M3_MIN_SUBNORMAL = 2.0 ** -9
FP8_E4M3_MAX_FINITE = 240.0
_NATIVE_FP8_E4M3_NAMES = (
    "float8_e4m3fn",
    "float8_e4m3",
    "float8e4m3fn",
    "float8e4m3",
)


def native_fp8_e4m3_dtype() -> Optional[mx.Dtype]:
    """Return a native MLX float8 E4M3 dtype if the runtime exposes one."""
    for name in _NATIVE_FP8_E4M3_NAMES:
        if hasattr(mx, name):
            return getattr(mx, name)
    return None


def pack_fp8_e4m3(tensor: mx.array) -> mx.array:
    """Pack a tensor into one-byte software E4M3 values.

    The software format is finite-only and saturates values outside the
    normal representable range to the largest finite code.
    """
    x = tensor.astype(mx.float32)
    x = mx.where(x == x, x, 0.0)

    sign_bits = (x < 0).astype(mx.int32) * 128
    magnitude = mx.minimum(mx.abs(x), FP8_E4M3_MAX_FINITE)

    zero_mask = magnitude == 0.0
    normal_mask = magnitude >= FP8_E4M3_MIN_NORMAL
    subnormal_mask = (~zero_mask) & (~normal_mask)

    safe_magnitude = mx.maximum(magnitude, FP8_E4M3_MIN_SUBNORMAL)
    exponent = mx.floor(mx.log2(safe_magnitude)).astype(mx.int32)
    exponent = mx.clip(exponent, -6, 7)
    exponent_scale = mx.power(2.0, exponent.astype(mx.float32))
    normalized = magnitude / exponent_scale

    mantissa = mx.round((normalized - 1.0) * 8.0).astype(mx.int32)
    carry = mantissa >= 8
    mantissa = mx.where(carry, 0, mantissa)
    exponent = exponent + carry.astype(mx.int32)

    overflow = exponent > 7
    exponent_bits = mx.where(overflow, 14, exponent + FP8_E4M3_EXPONENT_BIAS)
    mantissa = mx.where(overflow, 7, mantissa)
    normal_bits = sign_bits + exponent_bits * 8 + mantissa

    subnormal_mantissa = mx.round(magnitude / FP8_E4M3_MIN_SUBNORMAL).astype(mx.int32)
    subnormal_mantissa = mx.clip(subnormal_mantissa, 0, 7)
    subnormal_bits = sign_bits + subnormal_mantissa

    packed = mx.where(normal_mask, normal_bits, mx.where(subnormal_mask, subnormal_bits, 0))
    return packed.astype(mx.uint8)


def unpack_fp8_e4m3(tensor: mx.array, dtype: mx.Dtype = mx.float16) -> mx.array:
    """Unpack software E4M3 bytes into a standard MLX floating-point dtype."""
    bits = tensor.astype(mx.int32)
    payload = bits % 128
    sign = mx.where(bits >= 128, -1.0, 1.0)

    exponent_bits = payload // 8
    mantissa_bits = payload % 8

    normal_mask = exponent_bits > 0
    subnormal_mask = (exponent_bits == 0) & (mantissa_bits > 0)

    normal = sign * (1.0 + mantissa_bits.astype(mx.float32) / 8.0)
    normal = normal * mx.power(2.0, exponent_bits.astype(mx.float32) - FP8_E4M3_EXPONENT_BIAS)

    subnormal = sign * mantissa_bits.astype(mx.float32) * FP8_E4M3_MIN_SUBNORMAL
    unpacked = mx.where(normal_mask, normal, mx.where(subnormal_mask, subnormal, 0.0))
    return unpacked.astype(dtype)
