"""Block-Attention GPUModelRunner.

Identical to the lazy runner except that documents are re-positioned in the
cached *keys* (in the runner, before the attention kernel) the literal
Block-Attention way -- un-rotate each key to position 0, then rotate to its
target contiguous position -- instead of rotating the query per-block inside the
kernel. After placing K we neutralise the kernel's Q-rotation by setting the
request's q_offset to 1 (q_mask is preserved) and re-refreshing the metadata
buffers, so a standard attention kernel runs over keys that already sit at their
target absolute positions.

See [[block-attn-vllm-implementation]].

Changed by Haocheng at 2025/09/08; rotation implemented 2026-05-28.
"""

from typing import TYPE_CHECKING

import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.config import VllmConfig
from vllm.logger import init_logger

import os

from lazy.worker.gpu_model_runner import LazyGPUModelRunner
from block_attn_vllm.utils import (block_key_norm, copy_paged_block,
                                    place_paged_block_keys)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class BlockAttnGPUModelRunner(LazyGPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._cos_sin_cache: torch.Tensor | None = None

    def _model_cos_sin_cache(self) -> torch.Tensor:
        """The model's own RoPE cos/sin cache (so rotations match exactly)."""
        if self._cos_sin_cache is None:
            for module in self.model.modules():
                cache = getattr(module, "cos_sin_cache", None)
                if cache is not None:
                    self._cos_sin_cache = cache
                    break
            assert self._cos_sin_cache is not None, (
                "could not find a rotary embedding cos_sin_cache on the model")
        return self._cos_sin_cache

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        super()._update_states(scheduler_output)
        self._apply_block_rotation(scheduler_output)

    def _apply_block_rotation(self, scheduler_output: "SchedulerOutput") -> None:
        """Copy each document block (copy-on-write) and place it, the literal
        Block-Attention way.

        The scheduler redirected each lazy request's document blocks to fresh
        copies and recorded `cow_blocks` = (new_id, old_id, from_start,
        to_start). For every layer we copy the canonical KV into the copy, then
        re-position the copy's keys by un-rotating each token from its stored
        local position (from_start + slot) back to 0 and rotating it to its
        target contiguous position (to_start + slot). The canonical offset-0
        blocks are never touched, so they stay re-usable by the prefix cache.

        Finally we neutralise the kernel's Q-rotation (q_offset = 1, q_mask
        preserved) AND re-refresh the lazy metadata buffers: super()._update_
        states() already copied the original q_offset into self.lazy_offset
        before this runs, so without the re-refresh the kernel would rotate Q by
        R(-abs) on top of our K placement -- a doubled offset that corrupts the
        output.
        """
        new_reqs = [r for r in scheduler_output.scheduled_new_reqs
                    if getattr(r, "is_lazy", False) and r.q_offset is not None]
        if not new_reqs or not self.kv_caches:
            return

        hf = self.model_config.hf_config
        num_kv_heads = hf.num_key_value_heads
        head_dim = (getattr(hf, "head_dim", None)
                    or hf.hidden_size // hf.num_attention_heads)
        dtype = self.kv_caches[0].dtype
        cos_sin_cache = self._model_cos_sin_cache()
        debug = bool(os.environ.get("BLOCK_ATTN_DEBUG"))

        split = PagedAttention.split_kv_cache
        for new_req in new_reqs:
            cow_blocks = getattr(new_req, "cow_blocks", None)
            if cow_blocks:
                for layer_idx, kv in enumerate(self.kv_caches):
                    key_cache, value_cache = split(kv, num_kv_heads, head_dim)
                    for new_id, old_id, from_start, to_start in cow_blocks:
                        if debug and layer_idx == 0:
                            logger.info(
                                "[BA] req=%s doc block old=%d new=%d "
                                "from=%d to=%d src_norm=%.2f",
                                new_req.req_id, old_id, new_id, from_start,
                                to_start, block_key_norm(key_cache, old_id))
                        copy_paged_block(key_cache, value_cache, new_id, old_id)
                        place_paged_block_keys(key_cache, new_id, cos_sin_cache,
                                               from_start, to_start, dtype)

            # Neutralise the kernel Q-rotation; keep q_mask for padding.
            self.requests[new_req.req_id].q_offset = [1] * len(new_req.q_offset)

        # CRITICAL: repopulate the GPU metadata buffer so the rebuilt packed
        # block table sees q_offset == 1 (see docstring).
        self._refresh_lazy_metadata_buffers()
        self._packed_block_table_full_rebuild = True
        self._packed_block_table_delta_rows.clear()
