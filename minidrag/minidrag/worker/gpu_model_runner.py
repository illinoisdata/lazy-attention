""" Modified from vllm/v1/worker/gpu_model_runner.py 

This version is modified to support dynamic RAG requests. Make the following changes:

1. _update_states(): to enable the state update for document processing which does not exist in the original code.
2. _prepare_inputs(): to enable the document cutting.
"""

# SPDX-License-Identifier: Apache-2.0

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
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import get_pp_group, graph_capture
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, LayerBlockType, LazyLoader, cdiv,
                        check_use_alibi, is_pin_memory_available)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from .utils import sanity_check_mm_encoder_outputs

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


# class GPUModelRunner(LoRAModelRunnerMixin):
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
            prompt=new_req_data.prompt,
            mm_inputs=new_req_data.mm_inputs,
            mm_positions=new_req_data.mm_positions,
            sampling_params=sampling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            image_grid_thw = []
            video_grid_thw = []
            second_per_grid_ts = []
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

            hf_config = self.model_config.hf_config

            self.requests[req_id].mrope_positions, \
                self.requests[req_id].mrope_position_delta = \
                MRotaryEmbedding.get_input_positions_tensor(
                    self.requests[req_id].prompt_token_ids,
                    hf_config=hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
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
    removed_req_indices = sorted(removed_req_indices, reverse=True)
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

    if batch_changed:
        self.input_batch.refresh_sampling_metadata()

def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
) -> tuple[FlashAttentionMetadata, torch.Tensor,
           Optional[SpecDecodeMetadata]]:
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # Some attention backends (namely MLA) may want to separate requests
    # based on if the attention computation will be compute-bound or
    # memory-bound. This gives them a hook to do that.
    modified_batch = self.attn_metadata_builder.reorder_batch(
        self.input_batch, scheduler_output)
    if modified_batch:
        self.input_batch.refresh_sampling_metadata()

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit(num_reqs)

    # Get the number of scheduled tokens for each request.
    # TODO: The Python loop can be slow. Optimize.
    num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
    max_num_scheduled_tokens = 0
    for i, req_id in enumerate(self.input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens[i] = num_tokens
        max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                       num_tokens)

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
    # NOTE(woosuk): We use torch.index_select instead of np.take here
    # because torch.index_select is much faster than np.take for large
    # tensors.
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

    attn_metadata = self.attn_metadata_builder.build(
        num_reqs=num_reqs,
        num_actual_tokens=total_num_scheduled_tokens,
        max_query_len=max_num_scheduled_tokens,
        common_prefix_len=common_prefix_len,
    )

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

    return attn_metadata, logits_indices, spec_decode_metadata

@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, torch.Tensor]:
    self._update_states(scheduler_output)
    if not scheduler_output.total_num_scheduled_tokens:
        # Return empty ModelRunnerOuptut if there's no work to do.
        return EMPTY_MODEL_RUNNER_OUTPUT

    if self.is_multimodal_model:
        # Run the multimodal encoder if any.
        self._execute_encoder(scheduler_output)
        encoder_outputs = self._gather_encoder_outputs(scheduler_output)
    else:
        encoder_outputs = []

    # Prepare the decoder inputs.
    attn_metadata, logits_indices, spec_decode_metadata = (
        self._prepare_inputs(scheduler_output))
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    if (self.use_cuda_graph
            and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
        # Use piecewise CUDA graphs.
        # Add padding to the batch size.
        num_input_tokens = self.vllm_config.pad_for_cudagraph(
            num_scheduled_tokens)
    else:
        # Eager mode.
        num_input_tokens = num_scheduled_tokens
    attn_metadata.num_input_tokens = num_input_tokens

    if self.is_multimodal_model:
        # NOTE(woosuk): To unify token ids and soft tokens (vision
        # embeddings), we always use embeddings (rather than token ids)
        # as input to the multimodal model, even when the input is text.
        input_ids = self.input_ids[:num_scheduled_tokens]
        if encoder_outputs:
            inputs_embeds = self.model.get_input_embeddings(
                input_ids, encoder_outputs)
        else:
            inputs_embeds = self.model.get_input_embeddings(input_ids)
        # TODO(woosuk): Avoid the copy. Optimize.
        self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
        inputs_embeds = self.inputs_embeds[:num_input_tokens]
        input_ids = None
    else:
        # For text-only models, we use token ids as input.
        # While it is possible to use embeddings as input just like the
        # multimodal models, it is not desirable for performance since
        # then the embedding layer is not included in the CUDA graph.
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None
    if self.uses_mrope:
        positions = self.mrope_positions[:, :num_input_tokens]
    else:
        positions = self.positions[:num_input_tokens]

    if get_pp_group().is_first_rank:
        intermediate_tensors = None
    else:
        assert intermediate_tensors is not None
        assert self.intermediate_tensors is not None
        for k, v in intermediate_tensors.items():
            self.intermediate_tensors[k][:num_input_tokens].copy_(
                v[:num_input_tokens], non_blocking=True)
        intermediate_tensors = IntermediateTensors({
            k: v[:num_input_tokens]
            for k, v in self.intermediate_tensors.items()
        })

    # Run the decoder.
    # Use persistent buffers for CUDA graphs.
    with set_forward_context(attn_metadata, self.vllm_config):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
    if not get_pp_group().is_last_rank:
        # For mid-pipeline stages, return the hidden states.
        return hidden_states

    hidden_states = hidden_states[:num_scheduled_tokens]
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states, None)

    # Apply structured output bitmasks if present
    if scheduler_output.grammar_bitmask is not None:
        self.apply_grammar_bitmask(scheduler_output, logits)

    # Sample the next token and get logprobs if needed.
    sampling_metadata = self.input_batch.sampling_metadata
    if spec_decode_metadata is None:
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
    else:
        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        sampler_output = self.model.sample(
            logits=bonus_logits,
            sampling_metadata=sampling_metadata,
        )
        bonus_token_ids = sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        target_logits = logits[spec_decode_metadata.target_logits_indices]
        output_token_ids = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
        sampler_output.sampled_token_ids = output_token_ids

    # TODO(woosuk): The following loop can be slow since it iterates over
    # the requests one by one. Optimize.
    discard_sampled_tokens_req_indices = []
    for i, req_id in enumerate(self.input_batch.req_ids):
        req_state = self.requests[req_id]
        seq_len = (req_state.num_computed_tokens +
                   scheduler_output.num_scheduled_tokens[req_id])
        if seq_len < req_state.num_tokens:
            # Ignore the sampled token for partial prefills.
            # Rewind the generator state as if the token was not sampled.
            # This relies on cuda-specific torch-internal impl details
            generator = self.input_batch.generators.get(i)
            if generator is not None:
                generator.set_offset(generator.get_offset() - 4)
            # Record the index of the request that should not be sampled,
            # so that we could clear the sampled tokens before returning.
            discard_sampled_tokens_req_indices.append(i)

    # NOTE: GPU -> CPU Sync happens here.
    # Move as many CPU operations as possible before this sync point.
    logprobs_tensors = sampler_output.logprobs_tensors
    logprobs_lists = logprobs_tensors.tolists() \
        if logprobs_tensors is not None else None

    # Compute prompt logprobs if needed.
    prompt_logprobs_dict = self._get_prompt_logprobs_dict(
        hidden_states,
        scheduler_output,
    )

    # Get the valid generated tokens.
    sampled_token_ids = sampler_output.sampled_token_ids
    max_gen_len = sampled_token_ids.shape[-1]
    if max_gen_len == 1:
        # No spec decode tokens.
        valid_sampled_token_ids = sampled_token_ids.tolist()
    else:
        # Includes spec decode tokens.
        valid_sampled_token_ids = self.rejection_sampler.parse_output(
            sampled_token_ids,
            self.input_batch.vocab_size,
        )
    # Mask out the sampled tokens that should not be sampled.
    for i in discard_sampled_tokens_req_indices:
        valid_sampled_token_ids[i].clear()

    if not self.use_spec_decode:
        # Speculative decoding is not enabled.
        spec_token_ids = None
    elif self.speculative_config.method == "ngram":
        assert isinstance(self.drafter, NgramProposer)
        spec_token_ids = self.generate_draft_token_ids(
            valid_sampled_token_ids, sampling_metadata)
    elif self.speculative_config.method == "eagle":
        assert isinstance(self.drafter, EagleProposer)
        # TODO(woosuk): Refactor the loop.
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(valid_sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = self.input_batch.req_ids[i]
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(next_token_ids,
                                      dtype=torch.int32,
                                      device=self.device)

        if spec_decode_metadata is None:
            # input_ids can be None for multimodal models.
            target_token_ids = self.input_ids[:num_scheduled_tokens]
            target_positions = positions
            target_hidden_states = hidden_states
            target_slot_mapping = attn_metadata.slot_mapping
            cu_num_tokens = attn_metadata.query_start_loc
        else:
            # TODO(woosuk): Refactor this.
            num_draft_tokens = spec_decode_metadata.num_draft_tokens
            num_rejected_tokens = [
                n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                for i, n in enumerate(num_draft_tokens)
            ]
            num_rejected_tokens = torch.tensor(
                num_rejected_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            cu_num_tokens, token_indices = self.drafter.prepare_inputs(
                attn_metadata.query_start_loc,
                num_rejected_tokens,
            )
            target_token_ids = self.input_ids[token_indices]
            target_positions = positions[token_indices]
            target_hidden_states = hidden_states[token_indices]
            target_slot_mapping = attn_metadata.slot_mapping[token_indices]

        draft_token_ids, draft_probs = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            target_slot_mapping=target_slot_mapping,
            next_token_ids=next_token_ids,
            cu_num_tokens=cu_num_tokens,
            block_table=attn_metadata.block_table,
            sampling_metadata=sampling_metadata,
        )
        spec_token_ids = draft_token_ids.tolist()
        # TODO(woosuk): Cache draft_probs and use it for rejection sampling
        # in the next step.
        del draft_probs

    return ModelRunnerOutput(
        req_ids=self.input_batch.req_ids,
        req_id_to_index=self.input_batch.req_id_to_index,
        sampled_token_ids=valid_sampled_token_ids,
        spec_token_ids=spec_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
    )
