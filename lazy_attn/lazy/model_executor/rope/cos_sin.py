"""In-kernel RoPE cos/sin device functions (decode path) + test mirrors.

A `@triton.jit` device function returns cos/sin for one position in the NeoX
duplicated layout `[HEAD_SIZE]` (first/second half identical), matching the
layout the decode kernel expects (and the cos_sin_cache rows it replaces).

`rope_cos_sin` dispatches by a constexpr ROPE_TYPE so one shared decode kernel
can serve different models; the scaling parameters come from the model's rotary
layer (`rope_meta_from_layer`), so the result equals that layer's cos_sin_cache.
"""

from __future__ import annotations

import math

import triton
import triton.language as tl
from triton.language.extra import libdevice

ROPE_DEFAULT: int = 0
ROPE_LLAMA3: int = 1


@triton.jit
def rope_freqs(
    ROPE_TYPE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BASE: tl.constexpr,
    SCALING_FACTOR: tl.constexpr,
    LOW_FACTOR: tl.constexpr,
    HIGH_FACTOR: tl.constexpr,
    ORIG_MAX_POSITION: tl.constexpr,
    PI_VALUE: tl.constexpr,
):
    """The position-*independent* half of the RoPE formula: [HEAD_SIZE] freqs.

    Split out of `rope_cos_sin` so a caller inside a loop can evaluate the
    expensive part -- `pow`, and the Llama-3 smoothing on top of it -- once,
    and keep only `cos/sin(position * freqs)` in the loop body.
    """
    half: tl.constexpr = HEAD_SIZE // 2
    inv_freqs = 1.0 / libdevice.pow(
        BASE, 2 * (tl.arange(0, HEAD_SIZE) % half) / HEAD_SIZE)
    if ROPE_TYPE == 1:  # ROPE_LLAMA3
        low_freq_wavelen: tl.constexpr = ORIG_MAX_POSITION / LOW_FACTOR
        high_freq_wavelen: tl.constexpr = ORIG_MAX_POSITION / HIGH_FACTOR
        wave_len = 2 * PI_VALUE / inv_freqs
        smooth = (ORIG_MAX_POSITION / wave_len - LOW_FACTOR) / (
            HIGH_FACTOR - LOW_FACTOR)
        return tl.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            tl.where(
                wave_len > low_freq_wavelen,
                inv_freqs / SCALING_FACTOR,
                (1 - smooth) * inv_freqs / SCALING_FACTOR + smooth * inv_freqs,
            ),
        )
    return inv_freqs


@triton.jit
def rope_cos_sin_from_freqs(position, freqs):
    """cos/sin for one position, given precomputed `rope_freqs` output."""
    theta = position * freqs
    return libdevice.cos(theta), libdevice.sin(theta)


@triton.jit
def rope_cos_sin(
    position,
    ROPE_TYPE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BASE: tl.constexpr,
    SCALING_FACTOR: tl.constexpr,
    LOW_FACTOR: tl.constexpr,
    HIGH_FACTOR: tl.constexpr,
    ORIG_MAX_POSITION: tl.constexpr,
    PI_VALUE: tl.constexpr,
):
    """cos/sin for one position, for callers outside a loop.

    Equivalent to `rope_cos_sin_from_freqs(position, rope_freqs(...))`. Inside a
    loop, call the two halves separately so the frequency table is built once --
    that is what the decode kernels do (see the register measurements in
    docs/design.md 4.3).
    """
    return rope_cos_sin_from_freqs(
        position,
        rope_freqs(ROPE_TYPE, HEAD_SIZE, BASE, SCALING_FACTOR, LOW_FACTOR,
                   HIGH_FACTOR, ORIG_MAX_POSITION, PI_VALUE))


def rope_meta_from_layer(rotary_emb) -> dict:
    """Constexpr RoPE parameters sourced from the model's rotary layer.

    Returns keys ROPE_TYPE, BASE, SCALING_FACTOR, LOW_FACTOR, HIGH_FACTOR,
    ORIG_MAX_POSITION, PI_VALUE -- passed straight to the decode kernel as
    constexpr. HEAD_SIZE is already supplied to the kernel separately.
    """
    base = float(getattr(rotary_emb, "base", 10000.0))
    pi = math.pi
    if (hasattr(rotary_emb, "scaling_factor")
            and hasattr(rotary_emb, "low_freq_factor")
            and hasattr(rotary_emb, "high_freq_factor")
            and hasattr(rotary_emb, "orig_max_position")):
        return dict(
            ROPE_TYPE=ROPE_LLAMA3,
            BASE=base,
            SCALING_FACTOR=float(rotary_emb.scaling_factor),
            LOW_FACTOR=float(rotary_emb.low_freq_factor),
            HIGH_FACTOR=float(rotary_emb.high_freq_factor),
            ORIG_MAX_POSITION=int(rotary_emb.orig_max_position),
            PI_VALUE=pi,
        )
    return dict(
        ROPE_TYPE=ROPE_DEFAULT,
        BASE=base,
        SCALING_FACTOR=1.0,
        LOW_FACTOR=1.0,
        HIGH_FACTOR=1.0,
        ORIG_MAX_POSITION=8192,
        PI_VALUE=pi,
    )


def llama3_cos_sin_torch(position: int, head_size: int, base: float,
                         scaling_factor: float, low_factor: float,
                         high_factor: float, orig_max_position: int, device):
    """Torch mirror of `llama3_cos_sin` for parity testing vs cos_sin_cache.

    Returns (cos, sin) each shape [head_size] (NeoX duplicated).
    """
    import torch
    half = head_size // 2
    idx = (torch.arange(head_size, device=device) % half).float()
    inv = 1.0 / (base ** (2 * idx / head_size))
    wave_len = 2 * math.pi / inv
    low_wl = orig_max_position / low_factor
    high_wl = orig_max_position / high_factor
    smooth = (orig_max_position / wave_len - low_factor) / (
        high_factor - low_factor)
    new = torch.where(
        wave_len < high_wl, inv,
        torch.where(wave_len > low_wl, inv / scaling_factor,
                    (1 - smooth) * inv / scaling_factor + smooth * inv))
    theta = position * new
    return torch.cos(theta), torch.sin(theta)
