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
