"""
Lazy attention version of GPUModelRunner.

Changed by Haocheng at 2025/09/08
"""

import gc
import time
import weakref
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import get_pp_group, graph_capture
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, LayerBlockType, LazyLoader, cdiv,
                        check_use_alibi, is_pin_memory_available)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)

# ////////////////////////////////////
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from lazy.worker.gpu_input_batch import CachedRequestState
from lazy.attention.backends.flash_attn import FlashAttentionMetadata
from lazy.utils.variants import (lazy_shared_kv_profile_enabled,
                                 lazy_shared_kv_profile_min_reqs,
                                 lazy_packed_block_profile_enabled)

LAZY_SHARED_KV_PROFILE = lazy_shared_kv_profile_enabled()
LAZY_SHARED_KV_PROFILE_MIN_REQS = lazy_shared_kv_profile_min_reqs()
LAZY_PACKED_BLOCK_PROFILE = lazy_packed_block_profile_enabled()

class LazyGPUModelRunner(GPUModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)

        # vLLM 0.9.x moved block_size / max_num_blocks_per_req onto the
        # per-group BlockTable (block tables became multi-group). The lazy
        # side-buffers below are indexed like the attention block table, so
        # recompute the same bounds the block table uses for this model.
        self.block_size = self.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)

        # ///////////////////////////
        self.is_lazy_req = torch.zeros(self.max_num_reqs,
                                       dtype=torch.bool,
                                       device=self.device)
        self.is_lazy_req_cpu = torch.zeros(self.max_num_reqs,
                                           dtype=torch.bool,
                                           device="cpu",
                                           pin_memory=self.pin_memory)
        self.is_lazy_req_np = self.is_lazy_req_cpu.numpy()

        self.lazy_variant = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device=self.device)
        self.lazy_variant_cpu = torch.zeros(self.max_num_reqs,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.lazy_variant_np = self.lazy_variant_cpu.numpy()
        
        self.lazy_offset = torch.zeros((self.max_num_reqs,
                                        self.max_num_blocks_per_req),
                                       dtype=torch.int32,
                                       device=self.device)
        self.lazy_offset_cpu = torch.zeros((self.max_num_reqs, 
                                            self.max_num_blocks_per_req),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.lazy_offset_np = self.lazy_offset_cpu.numpy()
        
        self.lazy_mask = torch.zeros((self.max_num_reqs,
                                        self.max_num_blocks_per_req),
                                       dtype=torch.int32,
                                       device=self.device)
        self.lazy_mask_cpu = torch.zeros((self.max_num_reqs, 
                                            self.max_num_blocks_per_req),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.lazy_mask_np = self.lazy_mask_cpu.numpy()

        # Persistent packed block tables for the decode kernels, one per KV
        # cache group. Layout: [physical_block_idx:32 | q_offset:16 | q_mask:16]
        #
        # One table is not enough: block ids are only meaningful within the
        # group that allocated them, so a layer in group 1 handed group 0's
        # ids would read another group's cache. The real table set is built in
        # `initialize_kv_cache`, once the groups are known; this single-group
        # default covers anything that touches the buffers before then.
        self.packed_block_tables = [
            torch.zeros(
                (self.max_num_reqs, self.max_num_blocks_per_req),
                dtype=torch.int64,
                device=self.device,
            )
        ]
        self._layer_to_kv_group: dict[str, int] = {}
        self._packed_block_table_full_rebuild = True
        self._packed_block_table_delta_rows: set[int] = set()
        self._packed_block_counts = np.zeros((1, self.max_num_reqs),
                                             dtype=np.int64)
        # Snapshot of the batch composition, used to detect batch changes and
        # attention-backend reordering without relying on vLLM internals.
        self._last_req_ids: tuple = ()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)

        groups = kv_cache_config.kv_cache_groups
        # The rotation metadata is indexed by block, at the block size the
        # scheduler used to build it. A group on a different block size would
        # silently mis-map q_offset/q_mask onto its blocks, so refuse rather
        # than compute the wrong attention.
        odd = sorted({
            group.kv_cache_spec.block_size
            for group in groups
            if group.kv_cache_spec.block_size != self.block_size
        })
        if odd:
            raise NotImplementedError(
                "LazyAttention requires every KV cache group to use the "
                f"scheduler's block size ({self.block_size}); found "
                f"{odd}. The per-block rotation metadata is built at one "
                "block size and cannot address groups that differ.")

        block_tables = self.input_batch.block_table
        self.packed_block_tables = [
            torch.zeros(
                (self.max_num_reqs, block_tables[i].max_num_blocks_per_req),
                dtype=torch.int64,
                device=self.device,
            ) for i in range(len(groups))
        ]
        self._layer_to_kv_group = {
            layer_name: group_idx
            for group_idx, group in enumerate(groups)
            for layer_name in group.layer_names
        }
        self._packed_block_counts = np.zeros((len(groups), self.max_num_reqs),
                                             dtype=np.int64)
        self._packed_block_table_full_rebuild = True
        self._packed_block_table_delta_rows.clear()

    def _refresh_lazy_metadata_buffers(self) -> None:
        req_ids = self.input_batch.req_ids
        num_reqs = len(req_ids)

        self.is_lazy_req_cpu[:num_reqs].fill_(False)
        self.lazy_variant_cpu[:num_reqs].fill_(0)
        self.lazy_offset_cpu[:num_reqs].fill_(0)
        self.lazy_mask_cpu[:num_reqs].fill_(0)

        any_is_lazy = False
        for idx, req_id in enumerate(req_ids):
            req_state = self.requests[req_id]
            if not req_state.is_lazy:
                continue
            any_is_lazy = True
            self.is_lazy_req_cpu[idx] = True
            self.lazy_variant_np[idx] = req_state.lazy_variant
            if req_state.q_offset is not None:
                self.lazy_offset_np[idx, :len(req_state.q_offset)] = (
                    req_state.q_offset)
            if req_state.q_mask is not None:
                self.lazy_mask_np[idx, :len(req_state.q_mask)] = req_state.q_mask

        self.is_lazy_req[:num_reqs].copy_(self.is_lazy_req_cpu[:num_reqs],
                                          non_blocking=True)
        self.lazy_variant[:num_reqs].copy_(self.lazy_variant_cpu[:num_reqs],
                                           non_blocking=True)
        if any_is_lazy:
            self.lazy_offset[:num_reqs].copy_(self.lazy_offset_cpu[:num_reqs],
                                              non_blocking=True)
            self.lazy_mask[:num_reqs].copy_(self.lazy_mask_cpu[:num_reqs],
                                            non_blocking=True)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update cached states, then attach the lazy per-request metadata.

        The upstream implementation is large and changes every release, so we
        delegate to it and re-apply only the lazy-specific parts afterwards:

        1. copy the scheduler's per-request rotation metadata (q_offset/q_mask)
           onto the cached request state, and
        2. keep the packed block table bookkeeping in sync.

        `CachedRequestState` is itself patched to carry the lazy fields with
        defaults, so the base method can construct it without knowing about them.
        """
        super()._update_states(scheduler_output)

        # Attach lazy rotation metadata for newly scheduled requests.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_state = self.requests.get(new_req_data.req_id)
            if req_state is None:
                continue
            # They are all immutable for the lifetime of the request.
            req_state.is_lazy = getattr(new_req_data, "is_lazy", False)
            req_state.lazy_variant = getattr(new_req_data, "lazy_variant", 0)
            req_state.q_offset = getattr(new_req_data, "q_offset", None)
            req_state.q_mask = getattr(new_req_data, "q_mask", None)

        # A different req_id ordering covers additions, removals and
        # attention-backend reordering, without depending on base-class
        # internals that move between releases. The one case it cannot see is
        # an ID being reused -- abort `r0`, submit a new `r0` into the same
        # slot -- so any request arriving as new also forces a rebuild.
        req_ids_snapshot = tuple(self.input_batch.req_ids)
        if (scheduler_output.scheduled_new_reqs
                or req_ids_snapshot != self._last_req_ids):
            self._last_req_ids = req_ids_snapshot
            self._refresh_lazy_metadata_buffers()
            self._packed_block_table_full_rebuild = True
            self._packed_block_table_delta_rows.clear()

        # A row's packed entry is stale if *any* group grew it.
        num_reqs = self.input_batch.num_reqs
        block_tables = self.input_batch.block_table
        for group_idx in range(len(self.packed_block_tables)):
            current_block_counts = block_tables[group_idx].num_blocks_per_row
            if num_reqs != 0 and not self._packed_block_table_full_rebuild:
                changed_rows = np.flatnonzero(
                    self._packed_block_counts[group_idx, :num_reqs] !=
                    current_block_counts[:num_reqs])
                if changed_rows.size:
                    self._packed_block_table_delta_rows.update(
                        changed_rows.tolist())
            self._packed_block_counts[group_idx, :num_reqs] = (
                current_block_counts[:num_reqs])
            self._packed_block_counts[group_idx, num_reqs:] = 0

    def _rebuild_packed_block_table(self, num_reqs: int) -> None:
        """Refresh the packed block table consumed by the lazy decode kernels.

        Layout per entry: [physical_block_idx:32 | q_offset:16 | q_mask:16].
        """
        if not (self._packed_block_table_full_rebuild
                or self._packed_block_table_delta_rows):
            return

        packed_start = time.perf_counter() if LAZY_PACKED_BLOCK_PROFILE else 0.0
        full_rebuild = self._packed_block_table_full_rebuild
        delta_rows = () if full_rebuild else sorted(
            row for row in self._packed_block_table_delta_rows
            if row < num_reqs)

        rebuilt_rows = 0
        for group_idx, packed_table in enumerate(self.packed_block_tables):
            block_table_dev = (
                self.input_batch.block_table[group_idx].get_device_tensor())
            # q_offset/q_mask are per block and identical across groups (all
            # groups share the scheduler's block size, enforced in
            # initialize_kv_cache); only the block ids are group-specific.
            num_blocks = packed_table.shape[1]
            lazy_offset = self.lazy_offset[:, :num_blocks]
            lazy_mask = self.lazy_mask[:, :num_blocks]

            if full_rebuild:
                packed_block_table = packed_table[:num_reqs]
                packed_block_table.copy_(block_table_dev[:num_reqs],
                                         non_blocking=True)
                packed_block_table.bitwise_left_shift_(32)
                packed_block_table.bitwise_or_(
                    lazy_offset[:num_reqs].to(torch.int64) << 16)
                packed_block_table.bitwise_or_(
                    lazy_mask[:num_reqs].to(torch.int64))
                rebuilt_rows = num_reqs
            else:
                for row in delta_rows:
                    packed_row = packed_table[row]
                    packed_row.copy_(block_table_dev[row], non_blocking=True)
                    packed_row.bitwise_left_shift_(32)
                    packed_row.bitwise_or_(
                        lazy_offset[row].to(torch.int64) << 16)
                    packed_row.bitwise_or_(lazy_mask[row].to(torch.int64))
                rebuilt_rows = len(delta_rows)

        self._packed_block_table_full_rebuild = False
        self._packed_block_table_delta_rows.clear()

        if LAZY_PACKED_BLOCK_PROFILE:
            torch.cuda.synchronize()
            logger.info(
                "LazyPackedBlockProfile mode=%s groups=%d reqs=%d "
                "rebuilt_rows=%d elapsed_ms=%.3f",
                "full" if full_rebuild else "delta",
                len(self.packed_block_tables),
                num_reqs,
                rebuilt_rows,
                (time.perf_counter() - packed_start) * 1000.0,
            )

    def _attach_lazy_attn_metadata(self, attn_metadata, num_reqs: int) -> None:
        """Attach the lazy rotation tensors to the attention metadata.

        Since vLLM 0.9.x `_prepare_inputs` returns a layer-name -> metadata
        mapping (one entry per attention layer, often sharing one object per KV
        cache group). The rotation tensors are the same for every layer, but
        the packed block table is not: each layer must get the one built from
        *its* group's block ids.
        """
        if isinstance(attn_metadata, dict):
            entries = attn_metadata.items()
        else:
            entries = ((None, attn_metadata), )

        for layer_name, metadata in entries:
            if metadata is None:
                continue
            # We insert the lazy_mask and lazy_offset into the attn_metadata
            metadata.is_lazy = self.is_lazy_req[:num_reqs]
            metadata.lazy_variant = self.lazy_variant[:num_reqs]
            metadata.q_offset = self.lazy_offset[:num_reqs]
            metadata.q_mask = self.lazy_mask[:num_reqs]
            group_idx = self._layer_to_kv_group.get(layer_name, 0)
            metadata.packed_block_table = (
                self.packed_block_tables[group_idx][:num_reqs])

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ):
        """Delegate to vLLM, then layer the lazy metadata on top.

        The upstream body is rewritten most releases, so rather than forking it
        we run it and post-process: build the packed block table and attach the
        per-request rotation tensors to the attention metadata.
        """
        prepare_start = time.perf_counter() if LAZY_SHARED_KV_PROFILE else 0.0

        result = super()._prepare_inputs(scheduler_output)
        attn_metadata = result[0]

        num_reqs = self.input_batch.num_reqs
        self._rebuild_packed_block_table(num_reqs)
        self._attach_lazy_attn_metadata(attn_metadata, num_reqs)

        if LAZY_SHARED_KV_PROFILE and num_reqs >= LAZY_SHARED_KV_PROFILE_MIN_REQS:
            lazy_req_count = int(self.is_lazy_req_cpu[:num_reqs].sum())
            lazy_offset_blocks = 0
            for req_id in self.input_batch.req_ids:
                req_state = self.requests[req_id]
                if req_state.is_lazy and req_state.q_offset is not None:
                    lazy_offset_blocks += len(req_state.q_offset)
            logger.info(
                "LazyProfile prepare_inputs: reqs=%d scheduled_tokens=%d "
                "lazy_reqs=%d lazy_blocks=%d total_ms=%.3f",
                num_reqs,
                scheduler_output.total_num_scheduled_tokens,
                lazy_req_count,
                lazy_offset_blocks,
                (time.perf_counter() - prepare_start) * 1000.0,
            )

        return result
