"""Per-model RoPE cos/sin computed in-kernel (no cos_sin_cache table IO).

Design (per the lazy-attention roofline split):
  - Prefill is COMPUTE-bound -> the attention kernels LOAD cos/sin from the
    rotary layer's precomputed cos_sin_cache table (the load is hidden behind
    the big GEMMs).
  - Decode is IO/memory-bound -> the decode kernel COMPUTES cos/sin in-kernel
    from a handful of constexpr scaling parameters, avoiding the scattered
    cos_sin_cache table loads.

Each model's scaling lives in its own device function (dispatched by ROPE_TYPE),
and the scaling parameters are sourced from the model's actual rotary layer via
`rope_meta_from_layer` -- never hardcoded -- so the in-kernel cos/sin matches the
layer's cos_sin_cache field exactly (see debug/test_rope_cos_sin.py).
"""

from lazy.model_executor.rope.cos_sin import (
    ROPE_DEFAULT,
    ROPE_LLAMA3,
    default_cos_sin,
    llama3_cos_sin,
    llama3_cos_sin_torch,
    rope_cos_sin,
    rope_meta_from_layer,
)

__all__ = [
    "ROPE_DEFAULT",
    "ROPE_LLAMA3",
    "default_cos_sin",
    "llama3_cos_sin",
    "llama3_cos_sin_torch",
    "rope_cos_sin",
    "rope_meta_from_layer",
]
