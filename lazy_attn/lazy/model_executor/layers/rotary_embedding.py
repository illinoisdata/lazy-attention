"""Rotary Positional Embeddings for LazyAttention.

Changed by Haocheng at 2025/09/14
"""
from typing import Optional, Tuple

import torch

from vllm.model_executor.layers import rotary_embedding as vllm_rope
from vllm.model_executor.layers.rotary_embedding import _apply_rotary_emb_torch

from lazy.utils.variants import is_mepic_variant

# The variant is fixed for the process, so resolve it once at import.
USE_MEPIC_Q_ONLY_ROTARY = is_mepic_variant()


class _LazyRotaryMixin:
    """Shared behaviour for every RoPE class LazyAttention substitutes in.

    1. `inv_freq` is materialised as a buffer, because the model forward hands
       `self.rotary_emb.inv_freq` to the attention kernel while vLLM's base
       class keeps only the derived `cos_sin_cache`.
    2. Under the MEPIC variant the forward rotates the query only -- that
       kernel rotates the cached keys itself, so rotating them here as well
       would double-rotate them.

    Both hold whichever concrete RoPE class the checkpoint selects, hence the
    mixin: `rope_scaling: null` (the block-fine-tuned models) builds the plain
    `RotaryEmbedding`, not `Llama3RotaryEmbedding`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "inv_freq",
            self._compute_inv_freq(self.base).to(torch.bfloat16),
            persistent=False)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not USE_MEPIC_Q_ONLY_ROTARY:
            return super().forward_native(positions, query, key, offsets)

        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin,
                                            self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not USE_MEPIC_Q_ONLY_ROTARY:
            return super().forward_cuda(positions, query, key, offsets)

        from vllm import _custom_ops as ops

        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)

        # The op rotates in place and takes an *optional* key: passing None
        # skips the key half entirely, so the key reaches the attention kernel
        # unrotated without paying for a scratch copy of it every layer.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, None,
                                         self.head_size, self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, None, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key


class LazyRotaryEmbedding(_LazyRotaryMixin, vllm_rope.RotaryEmbedding):
    """Plain RoPE, for checkpoints with `rope_scaling: null`."""


class Llama3RotaryEmbedding(_LazyRotaryMixin, vllm_rope.Llama3RotaryEmbedding):
    """Llama-3 scaled RoPE.

    The frequency math stays upstream's; only `inv_freq` and the MEPIC forward
    come from the mixin. This subclasses the class object captured at import
    time, which is the original -- `vllm_patch` rebinds the module attribute
    afterwards.
    """
