import os

LAZY_VARIANT_NONE = 0
LAZY_VARIANT_LAZY = 1
LAZY_VARIANT_MEPIC = 2

_ENV_KEYS = ("LAZY_ATTENTION_VARIANT", "LAZY_ATTENTION_MODE", "VLLM_LAZY_VARIANT")
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def get_lazy_attention_variant_name() -> str:
    for key in _ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value.strip().lower()
    return "lazy"


def get_lazy_attention_variant_code() -> int:
    variant = get_lazy_attention_variant_name()
    if variant in ("lazy", "lazy_attn", "lazy_attention"):
        return LAZY_VARIANT_LAZY
    if variant in ("mepic", "nope", "position_independent"):
        return LAZY_VARIANT_MEPIC
    raise ValueError(
        f"Unsupported lazy attention variant '{variant}'. "
        "Expected one of: lazy, mepic."
    )


def mepic_first_block_recompute_enabled() -> bool:
    value = os.environ.get("MEPIC_FIRST_BLOCK_RECOMPUTE")
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def mepic_force_fp32_rotary_enabled() -> bool:
    value = os.environ.get("MEPIC_FORCE_FP32_ROTARY")
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def lazy_shared_kv_profile_enabled() -> bool:
    value = os.environ.get("LAZY_SHARED_KV_PROFILE")
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def lazy_shared_kv_profile_min_reqs() -> int:
    value = os.environ.get("LAZY_SHARED_KV_PROFILE_MIN_REQS")
    if value is None:
        return 32
    try:
        return max(int(value), 1)
    except ValueError:
        return 32


def lazy_packed_block_profile_enabled() -> bool:
    value = os.environ.get("LAZY_PACKED_BLOCK_PROFILE")
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES
