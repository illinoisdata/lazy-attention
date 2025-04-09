# SPDX-License-Identifier: Apache-2.0
"""Attention layer with PagedAttention and Triton prefix prefill."""
from typing import Any, Optional

import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata)

from ..ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode

# class TritonAttentionImpl(AttentionImpl):
def forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,  # key is not rotated when passing to this function
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: FlashAttentionMetadata,
    cos_sin_cache: Optional[torch.Tensor] = None,
    rotary_dim: Optional[int] = None,
    is_neox_style: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass impl with triton.
    Core attention computation is done in two functions called by `chunked_prefill_paged_decode`
    - `context_attention_fwd`
    - `kernel_paged_attention_2d`

    Note: We pass the `cos_sin_cache`, `rotary_dim`, and `is_neox_style` to the both for internal
    rotary embedding computation.

    Args:
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: Metadata for attention.
        cos_sin_cache: Cosine and sine cache for rotary embedding.
        rotary_dim: Dimension of rotary embedding.
        is_neox_style: Whether to use neox/gptj style rotary embedding.
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

    num_actual_tokens = attn_metadata.num_actual_tokens
    key_cache, value_cache = PagedAttention.split_kv_cache(
        kv_cache, self.num_kv_heads, self.head_size)

    # Reshape the input keys and values and store them in the cache.
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

    # Compute attention and update output up to `num_actual_tokens`.
    chunked_prefill_paged_decode(
        query=query[:num_actual_tokens],
        key=key[:num_actual_tokens],
        value=value[:num_actual_tokens],
        output=output[:num_actual_tokens],
        kv_cache_dtype=self.kv_cache_dtype,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=attn_metadata.block_table,
        query_start_loc=attn_metadata.query_start_loc,
        seq_lens=attn_metadata.seq_lens,
        max_seq_len=attn_metadata.max_seq_len,
        max_query_len=attn_metadata.max_query_len,
        k_scale=layer._k_scale,
        v_scale=layer._v_scale,
        alibi_slopes=self.alibi_slopes,
        sliding_window=self.sliding_window[0],
        sm_scale=self.scale,
        cos_sin_cache=cos_sin_cache,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )

    return output


original_forward = None

def apply_patch():
    import vllm.v1.attention.backends.triton_attn
    global original_forward
    original_forward = vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward
    vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward = forward


def revert_patch():
    import vllm.v1.attention.backends.triton_attn
    vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward = original_forward