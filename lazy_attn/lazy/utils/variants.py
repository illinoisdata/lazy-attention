"""The one place that decides which LazyAttention implementation runs.

Everything selectable at runtime is declared here, so switching variants is a
matter of reading this file rather than grepping for `os.environ` across the
kernels.

Attention variant -- `LAZY_ATTENTION_VARIANT` (aliases: `LAZY_ATTENTION_MODE`,
`VLLM_LAZY_VARIANT`):

    lazy   (default)  Deferred key rotation. RoPE rotates query and key
                      normally; the decode kernel re-rotates cached keys to
                      the slot they occupy.  -> lazy/attention/ops/
    mepic             Position-independent cache. RoPE rotates the query only,
                      because the kernel rotates the cached keys itself.
                      Aliases: `nope`, `position_independent`.
                      -> lazy/attention/old_ops/

A variant selects a matching kernel set, RoPE behaviour and scheduler rotation
metadata. The three consumers are:

    attention/backends/triton_attn.py            which kernel module to call
    model_executor/layers/rotary_embedding.py    rotate Q only, or Q and K
    core/sched/scheduler.py                      which q_offset/q_mask to emit

Tuning and profiling flags, boolean unless noted, accepting 1/true/yes/on:

    MEPIC_FIRST_BLOCK_RECOMPUTE     recompute each document's first block
                                    instead of reusing it (mepic only)
    MEPIC_FORCE_FP32_ROTARY         do the in-kernel rotation in fp32
    LAZY_SHARED_KV_PROFILE          log shared-KV scheduling stats
    LAZY_SHARED_KV_PROFILE_MIN_REQS int, default 32
    LAZY_PACKED_BLOCK_PROFILE       log packed block-table rebuild timings
    LAZY_PROFILE_ATTN_BACKEND       log per-layer attention timings

Decode-kernel switches, read per call so a benchmark can flip them between
runs. Each becomes a Triton constexpr, so a new value compiles a new kernel:

    NO_LAZY                       force every request onto the vanilla path
    LAZY_FORCE_SPLIT_DECODE       use the lazy-only decode kernel when every
                                  request in the batch is lazy
    LAZY_DECODE_IGNORE_Q_MASK     drop the document padding mask
    LAZY_DECODE_COMPUTE_COS_SIN   compute cos/sin in-kernel instead of loading
                                  them from cos_sin_cache; off by default,
                                  measured a win only for large-batch decode at
                                  head_size=128 (docs/design.md 4.3)
    LAZY_DECODE_WRAPPER_PROFILE   log decode wrapper timings
"""
import os

LAZY_VARIANT_LAZY = 1
LAZY_VARIANT_MEPIC = 2

_VARIANT_ENV_KEYS = ("LAZY_ATTENTION_VARIANT", "LAZY_ATTENTION_MODE",
                     "VLLM_LAZY_VARIANT")
_VARIANT_CODES = {
    "lazy": LAZY_VARIANT_LAZY,
    "lazy_attn": LAZY_VARIANT_LAZY,
    "lazy_attention": LAZY_VARIANT_LAZY,
    "mepic": LAZY_VARIANT_MEPIC,
    "nope": LAZY_VARIANT_MEPIC,
    "position_independent": LAZY_VARIANT_MEPIC,
}
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in _TRUTHY_ENV_VALUES


def get_lazy_attention_variant_name() -> str:
    for key in _VARIANT_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value.strip().lower()
    return "lazy"


def get_lazy_attention_variant_code() -> int:
    variant = get_lazy_attention_variant_name()
    try:
        return _VARIANT_CODES[variant]
    except KeyError:
        raise ValueError(
            f"Unsupported lazy attention variant '{variant}'. Expected one "
            f"of: {', '.join(sorted(_VARIANT_CODES))}.") from None


def is_mepic_variant() -> bool:
    return get_lazy_attention_variant_code() == LAZY_VARIANT_MEPIC


def mepic_first_block_recompute_enabled() -> bool:
    return _env_flag("MEPIC_FIRST_BLOCK_RECOMPUTE")


def mepic_force_fp32_rotary_enabled() -> bool:
    return _env_flag("MEPIC_FORCE_FP32_ROTARY")


def lazy_shared_kv_profile_enabled() -> bool:
    return _env_flag("LAZY_SHARED_KV_PROFILE")


def lazy_shared_kv_profile_min_reqs() -> int:
    value = os.environ.get("LAZY_SHARED_KV_PROFILE_MIN_REQS")
    if value is None:
        return 32
    try:
        return max(int(value), 1)
    except ValueError:
        return 32


def lazy_packed_block_profile_enabled() -> bool:
    return _env_flag("LAZY_PACKED_BLOCK_PROFILE")


def lazy_profile_attn_backend_enabled() -> bool:
    return _env_flag("LAZY_PROFILE_ATTN_BACKEND")


# Read per call -- see the module docstring.
def no_lazy_enabled() -> bool:
    return _env_flag("NO_LAZY")


def lazy_force_split_decode_enabled() -> bool:
    return _env_flag("LAZY_FORCE_SPLIT_DECODE")


def lazy_decode_ignore_q_mask_enabled() -> bool:
    return _env_flag("LAZY_DECODE_IGNORE_Q_MASK")


def lazy_decode_compute_cos_sin_enabled() -> bool:
    return _env_flag("LAZY_DECODE_COMPUTE_COS_SIN")


def lazy_decode_wrapper_profile_enabled() -> bool:
    return _env_flag("LAZY_DECODE_WRAPPER_PROFILE")
