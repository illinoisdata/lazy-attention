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
from vllm.v1.spec_decode.utils import is_spec_decode_supported
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

        # Persistent packed block table for decode kernels.
        # Layout: [physical_block_idx:32 | q_offset:16 | q_mask:16]
        self.packed_block_table = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int64,
            device=self.device,
        )
        self._packed_block_table_full_rebuild = True
        self._packed_block_table_delta_rows: set[int] = set()
        self._packed_block_counts = np.zeros(self.max_num_reqs, dtype=np.int64)

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
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
# /////////////////////////////////////////////////////////////////////////////
                # They are all immutable
                is_lazy=new_req_data.is_lazy,
                lazy_variant=new_req_data.lazy_variant,
                q_offset=new_req_data.q_offset,
                q_mask=new_req_data.q_mask,
# /////////////////////////////////////////////////////////////////////////////
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.extend(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices.sort(reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        # Some attention backends (namely MLA) may want to separate requests
        # based on if the attention computation will be compute-bound or
        # memory-bound. This gives them a hook to do that.
        batch_reordered = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)

        if batch_changed or batch_reordered:
            self._refresh_lazy_metadata_buffers()
            self.input_batch.refresh_sampling_metadata()
            self._packed_block_table_full_rebuild = True
            self._packed_block_table_delta_rows.clear()

        num_reqs = self.input_batch.num_reqs
        current_block_counts = self.input_batch.block_table.num_blocks_per_row
        if num_reqs != 0 and not self._packed_block_table_full_rebuild:
            changed_rows = np.flatnonzero(
                self._packed_block_counts[:num_reqs] != current_block_counts[:num_reqs]
            )
            if changed_rows.size:
                self._packed_block_table_delta_rows.update(changed_rows.tolist())
        self._packed_block_counts[:num_reqs] = current_block_counts[:num_reqs]
        self._packed_block_counts[num_reqs:] = 0

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[FlashAttentionMetadata, torch.Tensor,
               Optional[SpecDecodeMetadata]]:
        prepare_start = time.perf_counter() if LAZY_SHARED_KV_PROFILE else 0.0
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)
        
# /////////////////////////////////////////////////////////////////////////////
        lazy_meta_start = time.perf_counter() if LAZY_SHARED_KV_PROFILE else 0.0
        num_reqs = len(req_ids)
        any_is_lazy = bool(self.is_lazy_req_cpu[:num_reqs].any())
        lazy_req_count = int(self.is_lazy_req_cpu[:num_reqs].sum())
        lazy_offset_blocks = 0
        if any_is_lazy:
            for req_id in req_ids:
                req_state = self.requests[req_id]
                if req_state.is_lazy and req_state.q_offset is not None:
                    lazy_offset_blocks += len(req_state.q_offset)
        lazy_meta_elapsed_ms = (
            (time.perf_counter() - lazy_meta_start) * 1000.0
            if LAZY_SHARED_KV_PROFILE else 0.0
        )

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        # Prepare for cascade attention if enabled & beneficial.
        common_prefix_len = 0
        if self.cascade_attn_enabled:
            common_prefix_len = self._compute_cascade_attn_prefix_len(
                num_scheduled_tokens,
                scheduler_output.num_common_prefix_blocks,
            )

        if self._packed_block_table_full_rebuild or self._packed_block_table_delta_rows:
            packed_start = time.perf_counter() if LAZY_PACKED_BLOCK_PROFILE else 0.0
            block_table_dev = self.input_batch.block_table.get_device_tensor()
            if self._packed_block_table_full_rebuild:
                packed_block_table = self.packed_block_table[:num_reqs]
                packed_block_table.copy_(block_table_dev[:num_reqs], non_blocking=True)
                packed_block_table.bitwise_left_shift_(32)
                packed_block_table.bitwise_or_(
                    self.lazy_offset[:num_reqs].to(torch.int64) << 16)
                packed_block_table.bitwise_or_(
                    self.lazy_mask[:num_reqs].to(torch.int64))
                rebuilt_rows = num_reqs
                rebuild_mode = "full"
                self._packed_block_table_full_rebuild = False
                self._packed_block_table_delta_rows.clear()
            else:
                delta_rows = sorted(
                    row for row in self._packed_block_table_delta_rows if row < num_reqs
                )
                for row in delta_rows:
                    packed_row = self.packed_block_table[row]
                    packed_row.copy_(block_table_dev[row], non_blocking=True)
                    packed_row.bitwise_left_shift_(32)
                    packed_row.bitwise_or_(
                        self.lazy_offset[row].to(torch.int64) << 16)
                    packed_row.bitwise_or_(self.lazy_mask[row].to(torch.int64))
                rebuilt_rows = len(delta_rows)
                rebuild_mode = "delta"
                self._packed_block_table_delta_rows.clear()
            if LAZY_PACKED_BLOCK_PROFILE:
                torch.cuda.synchronize()
                logger.info(
                    "LazyPackedBlockProfile mode=%s reqs=%d rebuilt_rows=%d elapsed_ms=%.3f",
                    rebuild_mode,
                    num_reqs,
                    rebuilt_rows,
                    (time.perf_counter() - packed_start) * 1000.0,
                )

        attn_metadata = self.attn_metadata_builder.build(
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            common_prefix_len=common_prefix_len,
        )
# ///////////////////////////////////////////////////////////////////////////
        # We insert the lazy_mask and lazy_offset into the attn_metadata
        attn_metadata.is_lazy = self.is_lazy_req[:num_reqs]
        attn_metadata.lazy_variant = self.lazy_variant[:num_reqs]
        attn_metadata.q_offset = self.lazy_offset[:num_reqs]
        attn_metadata.q_mask = self.lazy_mask[:num_reqs]
        attn_metadata.packed_block_table = self.packed_block_table[:num_reqs]
# ////////////////////////////////////////////////////////////////////////////

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = attn_metadata.query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        if LAZY_SHARED_KV_PROFILE and num_reqs >= LAZY_SHARED_KV_PROFILE_MIN_REQS:
            logger.info(
                "LazyProfile prepare_inputs: reqs=%d scheduled_tokens=%d "
                "max_sched=%d any_is_lazy=%s lazy_reqs=%d lazy_blocks=%d "
                "lazy_meta_ms=%.3f total_ms=%.3f",
                num_reqs,
                total_num_scheduled_tokens,
                max_num_scheduled_tokens,
                any_is_lazy,
                lazy_req_count,
                lazy_offset_blocks,
                lazy_meta_elapsed_ms,
                (time.perf_counter() - prepare_start) * 1000.0,
            )

        return attn_metadata, logits_indices, spec_decode_metadata
