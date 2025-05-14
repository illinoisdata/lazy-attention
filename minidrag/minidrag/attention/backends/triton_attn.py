# SPDX-License-Identifier: Apache-2.0
"""Attention layer with PagedAttention and Triton prefix prefill."""
from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

from .flash_attn import FlashAttentionMetadata
from ..ops.triton_unified_attention import unified_attention

# class TritonAttentionImpl(AttentionImpl):
def forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: FlashAttentionMetadata,
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

    num_actual_tokens = attn_metadata.num_actual_tokens

    key_cache, value_cache = kv_cache.unbind(0)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        self.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )

    if self.kv_cache_dtype.startswith("fp8"):
        key_cache = key_cache.view(self.fp8_dtype)
        value_cache = value_cache.view(self.fp8_dtype)
        num_tokens, num_heads, head_size = query.shape
        assert layer._q_scale == 1.0, \
            "A non 1.0 q_scale is not currently supported."
        if not current_platform.is_rocm():
            # Skip Q quantization on ROCm, since dequantizing back to
            # f32 in the attention kernel is not supported.
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
        query = query.reshape((num_tokens, num_heads, head_size))

    use_local_attn = \
        (self.use_irope and attn_metadata.local_attn_metadata is not None)

    if use_local_attn:
        assert attn_metadata.local_attn_metadata is not None
        local_metadata = attn_metadata.local_attn_metadata
        cu_seqlens_q = local_metadata.local_query_start_loc
        seqused_k = local_metadata.local_seqused_k
        max_seqlen_q = local_metadata.local_max_query_len
        max_seqlen_k = local_metadata.local_max_seq_len
        block_table = local_metadata.local_block_table
        # ////
        is_lazy_req = local_metadata.local_is_lazy_req
        lazy_mask = local_metadata.local_lazy_mask
        lazy_offset = local_metadata.local_lazy_offset
    else:
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        # ////
        is_lazy_req = attn_metadata.is_lazy_req
        lazy_mask = attn_metadata.lazy_mask
        lazy_offset = attn_metadata.lazy_offset

    descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

    unified_attention(
        q=query[:num_actual_tokens],
        k=key_cache,
        v=value_cache,
        out=output[:num_actual_tokens],
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=self.scale,
        causal=True,
        alibi_slopes=self.alibi_slopes,
        window_size=self.sliding_window,
        block_table=block_table,
        softcap=self.logits_soft_cap,
        q_descale=None,  # Not supported
        k_descale=layer._k_scale.expand(descale_shape),
        v_descale=layer._v_scale.expand(descale_shape),
        cos_sin_cache=cos_sin_cache,
        is_lazy_req=is_lazy_req,
        lazy_mask=lazy_mask,
        lazy_offset=lazy_offset,
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