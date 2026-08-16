"""Per-model RoPE cos/sin computed in-kernel (an alternative to table IO).

The decode kernels need cos/sin for one rotation offset per distinct `q_offset`
in a sequence's block table. Two ways to get them:

  - **load** (default): index the rotary layer's precomputed `cos_sin_cache`.
  - **compute**: evaluate the RoPE formula in-kernel from a handful of constexpr
    scaling parameters, under `LAZY_DECODE_COMPUTE_COS_SIN=1`.

Compute trades memory traffic for registers, and which side wins depends on the
shape -- it is *not* a free win. See docs/design.md 4.3 for the measurements and
`benchmarks/bench_rope_cos_sin.py` to reproduce them.

The formula is split in two so a caller inside a loop pays the position-*inde-
pendent* half once: `rope_freqs` (the `pow`, plus the Llama-3 smoothing on top of
it, dispatched by a constexpr ROPE_TYPE) and `rope_cos_sin_from_freqs`
(`cos/sin(position * freqs)`). `rope_cos_sin` composes both for one-shot callers.

Scaling parameters are sourced from the model's actual rotary layer via
`rope_meta_from_layer` -- never hardcoded -- so the in-kernel cos/sin matches
that layer's `cos_sin_cache` field.
"""

from lazy.model_executor.rope.cos_sin import (
    ROPE_DEFAULT,
    ROPE_LLAMA3,
    llama3_cos_sin_torch,
    rope_cos_sin,
    rope_cos_sin_from_freqs,
    rope_freqs,
    rope_meta_from_layer,
)

__all__ = [
    "ROPE_DEFAULT",
    "ROPE_LLAMA3",
    "llama3_cos_sin_torch",
    "rope_cos_sin",
    "rope_cos_sin_from_freqs",
    "rope_freqs",
    "rope_meta_from_layer",
]
