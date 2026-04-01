# SPDX-License-Identifier: Apache-2.0
"""Attention layer with PagedAttention and Triton prefix prefill."""
import os
import time
from typing import Any, Optional

import torch

from vllm.attention.ops.paged_attn import PagedAttention
# from vllm.v1.attention.backends.flash_attn import (
#     FlashAttentionMetadata)

from .flash_attn import FlashAttentionMetadata
from ..ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode
from ..old_ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as chunked_prefill_paged_decode_old)
from lazy.utils.variants import (
    LAZY_VARIANT_MEPIC,
    get_lazy_attention_variant_code,
    mepic_force_fp32_rotary_enabled,
)


USE_OLD_MEPIC_KERNEL = (get_lazy_attention_variant_code() == LAZY_VARIANT_MEPIC)
MEPIC_FORCE_FP32_ROTARY = mepic_force_fp32_rotary_enabled()
PROFILE_ATTN_BACKEND = os.environ.get("LAZY_PROFILE_ATTN_BACKEND",
                                      "0") == "1"

_ATTN_PROFILE_STATE = {
    "calls": 0,
    "write_cache_ms": 0.0,
    "backend_ms": 0.0,
    "total_ms": 0.0,
    "per_layer": {},
}


def reset_attention_backend_profile() -> None:
    _ATTN_PROFILE_STATE["calls"] = 0
    _ATTN_PROFILE_STATE["write_cache_ms"] = 0.0
    _ATTN_PROFILE_STATE["backend_ms"] = 0.0
    _ATTN_PROFILE_STATE["total_ms"] = 0.0
    _ATTN_PROFILE_STATE["per_layer"] = {}


def get_attention_backend_profile() -> dict[str, float]:
    return {
        "calls": _ATTN_PROFILE_STATE["calls"],
        "write_cache_ms": _ATTN_PROFILE_STATE["write_cache_ms"],
        "backend_ms": _ATTN_PROFILE_STATE["backend_ms"],
        "total_ms": _ATTN_PROFILE_STATE["total_ms"],
        "per_layer": {
            layer: dict(stats)
            for layer, stats in _ATTN_PROFILE_STATE["per_layer"].items()
        },
    }

# class TritonAttentionImpl(AttentionImpl):
def forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: FlashAttentionMetadata,
    freqs: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    rotary_dim: Optional[int] = None,
    is_neox_style: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass with FlashAttention.
    Args:
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    """
    assert output is not None, "Output tensor must be provided."
    if attn_metadata is None:
        # Profiling run.
        return output
    assert attn_metadata.use_cascade is False
    # IMPORTANT!
    # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
    # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
    # in this method. For example, `view` and `slice` (or `[:n]`) operations
    # are surprisingly slow even in the case they do not invoke any GPU ops.
    # Minimize the PyTorch ops in this method as much as possible.
    # Whenever making a change in this method, please benchmark the
    # performance to make sure it does not introduce any overhead.
    total_start = time.perf_counter() if PROFILE_ATTN_BACKEND else 0.0
    num_actual_tokens = attn_metadata.num_actual_tokens
    key_cache, value_cache = PagedAttention.split_kv_cache(
        kv_cache, self.num_kv_heads, self.head_size)
    # Reshape the input keys and values and store them in the cache.
    stage_start = time.perf_counter() if PROFILE_ATTN_BACKEND else 0.0
    PagedAttention.write_to_paged_cache(
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        self.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )
    if PROFILE_ATTN_BACKEND:
        torch.cuda.synchronize()
        write_cache_ms = (time.perf_counter() - stage_start) * 1000.0
        _ATTN_PROFILE_STATE["write_cache_ms"] += (
            write_cache_ms)
    else:
        write_cache_ms = 0.0
    is_lazy = None
    lazy_variant = None
    q_offset = None
    q_mask = None
    use_local_attn = \
        (self.use_irope and attn_metadata.local_attn_metadata is not None)
    if use_local_attn:
        assert attn_metadata.local_attn_metadata is not None
        local_metadata = attn_metadata.local_attn_metadata
        cu_seqlens_q = local_metadata.local_query_start_loc
        sequesd_k = local_metadata.local_seqused_k
        max_seqlen_q = local_metadata.local_max_query_len
        max_seqlen_k = local_metadata.local_max_seq_len
        block_table = local_metadata.local_block_table
    else:
        cu_seqlens_q = attn_metadata.query_start_loc
        sequesd_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        packed_block_table = getattr(attn_metadata, "packed_block_table", None)
        # ////////
        is_lazy = attn_metadata.is_lazy
        lazy_variant = attn_metadata.lazy_variant
        q_offset = attn_metadata.q_offset
        q_mask = attn_metadata.q_mask
    # Compute attention and update output up to `num_actual_tokens`.
    
    # NOTE(haocheng): we give query a extra budget
    kernel_kwargs = dict(
        query=query[:num_actual_tokens+1],
        key=key[:num_actual_tokens],
        value=value[:num_actual_tokens],
        output=output[:num_actual_tokens],
        kv_cache_dtype=self.kv_cache_dtype,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=cu_seqlens_q,
        seq_lens=sequesd_k,
        max_seq_len=max_seqlen_k,
        max_query_len=max_seqlen_q,
        k_scale=layer._k_scale,
        v_scale=layer._v_scale,
        alibi_slopes=self.alibi_slopes,
        sliding_window=self.sliding_window[0],
        sm_scale=self.scale,
        cos_sin_cache=cos_sin_cache,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )
    stage_start = time.perf_counter() if PROFILE_ATTN_BACKEND else 0.0
    if USE_OLD_MEPIC_KERNEL:
        kernel_kwargs.update(
            q_offset=q_offset,
                q_mask=q_mask,
                is_mepic=True,
                force_fp32_rotary=MEPIC_FORCE_FP32_ROTARY,
            )
        chunked_prefill_paged_decode_old(**kernel_kwargs)
    else:
        kernel_kwargs.update(
            freqs=freqs,
            is_lazy=is_lazy,
            lazy_variant=lazy_variant,
            q_offset=q_offset,
            q_mask=q_mask,
            packed_block_table=packed_block_table,
        )
        chunked_prefill_paged_decode(**kernel_kwargs)
    if PROFILE_ATTN_BACKEND:
        torch.cuda.synchronize()
        backend_ms = (time.perf_counter() - stage_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        _ATTN_PROFILE_STATE["calls"] += 1
        _ATTN_PROFILE_STATE["backend_ms"] += backend_ms
        _ATTN_PROFILE_STATE["total_ms"] += total_ms
        layer_name = getattr(layer, "layer_name", "") or "<unknown>"
        layer_stats = _ATTN_PROFILE_STATE["per_layer"].setdefault(
            layer_name,
            {
                "calls": 0,
                "write_cache_ms": 0.0,
                "backend_ms": 0.0,
                "total_ms": 0.0,
            },
        )
        layer_stats["calls"] += 1
        layer_stats["write_cache_ms"] += write_cache_ms
        layer_stats["backend_ms"] += backend_ms
        layer_stats["total_ms"] += total_ms
    return output


# def forward(
#     self,
#     layer: torch.nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,  # key is not rotated when passing to this function
#     value: torch.Tensor,
#     kv_cache: torch.Tensor,
#     attn_metadata: FlashAttentionMetadata,
#     cos_sin_cache: Optional[torch.Tensor] = None,
#     rotary_dim: Optional[int] = None,
#     is_neox_style: bool = True,
#     output: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """Forward pass impl with triton.
#     Core attention computation is done in two functions called by `chunked_prefill_paged_decode`
#     - `context_attention_fwd`
#     - `kernel_paged_attention_2d`

#     Note: We pass the `cos_sin_cache`, `rotary_dim`, and `is_neox_style` to the both for internal
#     rotary embedding computation.

#     Args:
#         query: shape = [num_tokens, num_heads, head_size]
#         key: shape = [num_tokens, num_kv_heads, head_size]
#         value: shape = [num_tokens, num_kv_heads, head_size]
#         kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
#         attn_metadata: Metadata for attention.
#         cos_sin_cache: Cosine and sine cache for rotary embedding.
#         rotary_dim: Dimension of rotary embedding.
#         is_neox_style: Whether to use neox/gptj style rotary embedding.
#     Returns:
#         shape = [num_tokens, num_heads * head_size]
#     """
#     assert output is not None, "Output tensor must be provided."

#     if attn_metadata is None:
#         # Profiling run.
#         return output

#     assert attn_metadata.use_cascade is False

#     # IMPORTANT!
#     # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
#     # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
#     # in this method. For example, `view` and `slice` (or `[:n]`) operations
#     # are surprisingly slow even in the case they do not invoke any GPU ops.
#     # Minimize the PyTorch ops in this method as much as possible.
#     # Whenever making a change in this method, please benchmark the
#     # performance to make sure it does not introduce any overhead.

#     num_actual_tokens = attn_metadata.num_actual_tokens
#     key_cache, value_cache = PagedAttention.split_kv_cache(
#         kv_cache, self.num_kv_heads, self.head_size)

#     # Reshape the input keys and values and store them in the cache.
#     PagedAttention.write_to_paged_cache(
#         key,
#         value,
#         key_cache,
#         value_cache,
#         attn_metadata.slot_mapping,
#         self.kv_cache_dtype,
#         layer._k_scale,
#         layer._v_scale,
#     )

#     # Compute attention and update output up to `num_actual_tokens`.
#     chunked_prefill_paged_decode(
#         query=query[:num_actual_tokens],
#         key=key[:num_actual_tokens],
#         value=value[:num_actual_tokens],
#         output=output[:num_actual_tokens],
#         kv_cache_dtype=self.kv_cache_dtype,
#         key_cache=key_cache,
#         value_cache=value_cache,
#         block_table=attn_metadata.block_table,
#         query_start_loc=attn_metadata.query_start_loc,
#         seq_lens=attn_metadata.seq_lens,
#         max_seq_len=attn_metadata.max_seq_len,
#         max_query_len=attn_metadata.max_query_len,
#         k_scale=layer._k_scale,
#         v_scale=layer._v_scale,
#         alibi_slopes=self.alibi_slopes,
#         sliding_window=self.sliding_window[0],
#         sm_scale=self.scale,
#         cos_sin_cache=cos_sin_cache,
#         rotary_dim=rotary_dim,
#         is_neox_style=is_neox_style,
#     )

#     return output


original_forward = None

def apply_patch():
    import vllm.v1.attention.backends.triton_attn
    global original_forward
    original_forward = vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward
    vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward = forward


def revert_patch():
    import vllm.v1.attention.backends.triton_attn
    vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward = original_forward
