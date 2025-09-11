# SPDX-License-Identifier: Apache-2.0

"""
Core of Block Attention Scheduler

The logic is relatively simple.

1. We need to maintain an position array for each document
2. If documents are found in cache, position not matched and its ref cnt == 0, we can directly rotate,
   otherwise, we need to allocate new copies and copy these blocks, then rotate them

All logic is completed in BlockAttnScheduler

Changed by Haocheng at 2025/09/08
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

# Additional import for lazy attention
from itertools import chain
import copy
import numpy as np
from vllm.utils import cdiv, sha256

# Overwrite classes
from lazy.core.kv_cache_manager import LazyKVCacheManager
from lazy.request import RequestStatus
from lazy.request import LazyRequest as Request
from lazy.engine import EngineCoreRequest, EngineCoreEventType
from lazy.core.sched.output import NewRequestData

# For patch
from vllm.v1.core.sched.scheduler import Scheduler


class BlockAttnScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        logger.info("Initializing BlockAttnScheduler")
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.include_finished_set = include_finished_set

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            self.connector = KVConnectorFactory.create_connector_v1(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> deque of CachedRequestData
        self._cached_reqs_data: dict[
            str, deque[CachedRequestData]] = defaultdict(deque)

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = LazyKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats)
        
        # NOTE(Haocheng) 
        # Hyperparameters for scheduling. can affect the throughput.
        # - Maximum number of tokens to be processed in a single iteration.
        # - max_num_batched_tokens: int = field(default=None)  # type: ignore
        # - Maximum number of sequences to be processed in a single iteration.
        # e.g., max_num_seqs: int = 128
        logger.info(f"BlockAttnScheduler launched")
        
        # For block attention
        # NOTE(Haocheng): How does it work?
        # Block Attention will first discard all positional embeddings, and apply the new
        # positional embeddings based on the position in the block attention cache.
        # e.g., first the position is from 0 to 63, at this point, the offset is all 0 for involved blocks,
        # then when the position is from 64 to 127, the offset is all 64, etc.
        self.block_id_to_position_offset: dict[int, int] = {}

    def schedule(self) -> SchedulerOutput:
        logger.info(f"Scheduler: waiting={dump_dequeue(self.waiting)}, "
                    f"running={[req.request_id for req in self.running]}")
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        waiting_docs_reqs: dict[int, str] = {}  # (doc_hash, req_id)
        running_docs_reqs: dict[int, str] = {}  # (doc_hash, req_id)

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        
        # For lazy attention.
        req_to_q_offset: dict[str, list[int]] = {}
        req_to_q_mask: dict[str, list[int]] = {}

        # ///////////////////////////////////////////////////////////////////////
        # First, schedule the RUNNING requests.
        # NOTE(Haocheng):
        # If already in the RUNNING queue, the required documents are ready,
        # so we do not need to check the documents again.
        # TODO: clarify the ref cnt, is it possible that dep doc disappears
        # when inference, since we divide lazy req into multiple distinct reqs

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when PP>1 and
                #    we have already scheduled all prompt tokens but they are
                #    not finished yet.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                
                # NOTE(haocheng): if new_blocks are for documents, record it with offset 0
                if request.is_document_request:
                    for b in new_blocks:
                        self.block_id_to_position_offset[b.block_id] = 0
                
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()

        # ///////////////////////////////////////////////////////////////////////
        # Next, schedule the WAITING requests.
        # NOTE(Haocheng): Here we need to check and assemble query requests and 
        # documents.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

                # NOTE(Haocheng): check doc for lazy attention
                if request.has_documents:
                    # Check if documents are ready (ready means in the cache
                    # or fully scheduled, cache is allocated)
                    is_doc_ready_flags = self.is_doc_ready(request) # [num_docs，] bool
                    if not all(is_doc_ready_flags):
                        for doc_idx in range(len(is_doc_ready_flags)):
                            if not is_doc_ready_flags[doc_idx]:
                                # NOTE(haocheng): new spawned doc has higher priority
                                # than other waiting reqs, since it is blocking the
                                # query request
                                # DO NOT need to pop query req from waiting
                                self.add_doc_request(doc_idx, request)
                        continue

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens += num_external_tokens

                # NOTE(Haocheng): if one request reach here, there are three cases,
                # 1. it does not have documents
                #   1.1. it is a document request
                #   1.2. it is a normal request without documents (no extra process)
                # 2. its documents are all ready at *this* minor step

                # TODO(Haocheng): test whether this can bring benefit
                # fastening Case 1.1
                if request.is_document_request and \
                    num_computed_tokens == request.num_tokens:
                    self.waiting.popleft()
                    continue

                # NOTE(haocheng): new block is allocated, then we assemble
                # lazy request if needed
                # The role:
                # - General new request attends all documents
                # - Lock all documents by increasing ref cnt
                if request.has_documents:
                    # Case 2 -> Case 1.2
                    request.merge_documents()
                    logger.debug(f"Request {request.request_id} merges "
                                 f"documents, total prompt len "
                                 f"{request.num_prompt_tokens}")

                    computed_blocks_docs, num_computed_tokens_docs = \
                        self.kv_cache_manager.get_computed_blocks_docs(request)
                    computed_blocks = list(chain.from_iterable(
                        computed_blocks_docs)) + computed_blocks
                    assert sum(num_computed_tokens_docs) == \
                        sum(request.document_lens_padded), "Cached document lengths do not match"
                    num_computed_tokens += sum(num_computed_tokens_docs)
                    logger.debug(f"After merging documents, "
                                 f"request {request.request_id} has "
                                 f"{num_computed_tokens} computed tokens.")
                    
                    # NOTE(haocheng): !!! Core difference for block-attn from lazy attention !!!
                    # Here we decide copy, discard position, and re-assign position
                    # for all blocks in documents, we need to change their ref cnt
                    desired_position_offset = 0
                    begin_block_idx = 0
                    end_block_idx = 0
                    for doc_idx, blocks_for_one_doc in enumerate(computed_blocks_docs):
                        # Check if we need to copy
                        end_block_idx += len(blocks_for_one_doc)
                        need_to_copy = False
                        real_position_offset = self.block_id_to_position_offset.get(block.block_id, 0)
                        for block in blocks_for_one_doc:
                            if block.ref_cnt > 0 and real_position_offset != desired_position_offset:
                                need_to_copy = True
                                break
                            
                        if need_to_copy:
                            logger.info(f"Request {request.request_id} needs to copy document {doc_idx} since its position offset does not match desired offset {desired_position_offset}")
                            sampling_params = copy.deepcopy(request.sampling_params)
                            sampling_params.max_tokens = 1 # TODO(haocheng): how to avoid
                            doc_req = Request(
                                request_id=f"{request.request_id}_d{doc_idx}",
                                prompt_token_ids=request.documents_token_ids_padded[doc_idx],
                                multi_modal_inputs=request.mm_inputs,
                                multi_modal_hashes=request.mm_hashes,
                                multi_modal_placeholders=request.mm_positions,
                                sampling_params=sampling_params,
                                eos_token_id=request.eos_token_id,
                                is_document_request=True,
                                arrival_time=request.arrival_time,
                            )
                            new_blocks = self.kv_cache_manager.allocate_slots(request=doc_req, 
                                                                 num_new_tokens=self.block_size * len(blocks_for_one_doc))
                            assert new_blocks is not None and len(new_blocks) == len(blocks_for_one_doc), \
                                f"Allocated blocks {new_blocks} do not match old blocks {blocks_for_one_doc}"
                            
                            copy_blocks(from_block=blocks_for_one_doc, to_block=new_blocks)
                            reverse_rotate(new_blocks, real_position_offset)
                            rotate(new_blocks, desired_position_offset)
                            
                            # Replace the blocks for documents
                            computed_blocks[begin_block_idx:end_block_idx] = new_blocks
                            begin_block_idx = end_block_idx
                        # update desired position offset
                        desired_position_offset += self.block_size * len(blocks_for_one_doc)

                    # Get metadata for lazy attention
                    (req_to_q_offset[request.request_id], 
                     req_to_q_mask[request.request_id]) = \
                        metadata_for_lazy_attention(request, self.block_size)
                    logger.debug(f"Request {request.request_id} has "
                                 f"query offset {req_to_q_offset[request.request_id]} "
                                 f"and query mask {req_to_q_mask[request.request_id]}")
                    assert all([req_to_q_offset[request.request_id][i] == 0
                                for i in range(len(req_to_q_offset[request.request_id]))]), \
                        "After merging, the offset should be all 0"

                    # Update corresponding data in kv_cache_manager
                    # TODO(haocheng): optimize it
                    pre = self.kv_cache_manager.req_to_block_hashes_docs[request.request_id]
                    self.kv_cache_manager.req_to_block_hashes[request.request_id] = \
                        list(chain.from_iterable(pre)) + \
                        self.kv_cache_manager.req_to_block_hashes[request.request_id]

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    logger.debug(f"Long prefill req {request.request_id} "
                                 f"num_new_tokens {num_new_tokens} "
                                 f"threshold {self.scheduler_config.long_prefill_token_threshold}")
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_tokens,
                    computed_blocks, # will be touched in allocate_slots
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                
                # NOTE(haocheng): if new_blocks are for documents, record it with offset 0
                if request.is_document_request:
                    for b in new_blocks:
                        self.block_id_to_position_offset[b.block_id] = 0
                        
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVConnector: update internal state after allocation.
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        num_external_tokens,
                    )

                self.waiting.popleft()
                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # ///////////////////////////////////////////////////////////////////////
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running),
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id],
                                        req_to_q_offset.get(req.request_id, None),
                                        req_to_q_mask.get(req.request_id, None))
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        logger.info(f"Scheduler output: {scheduler_output}")
        return scheduler_output

    def add_request(self, request: Request, left=False) -> None:
        # NOTE(Haocheng): this function is used to add a request to the waiting
        # queue. For lazy attention, we add the request with `DOC_WAITING`
        tag = "[Normal]"
        if request.is_document_request:
            tag = "[Document]"
        elif request.has_documents:
            tag = "[Lazy]"
        logger.info(f"Adding {tag} request {request.request_id} to LazyScheduler")
        if left:
            self.waiting.appendleft(request)
        else:
            self.waiting.append(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def is_doc_ready(self, request: Request) -> bool:
        """Check if the documents are ready for the request."""
        assert request.has_documents
        _, num_computed_tokens_docs = \
            self.kv_cache_manager.get_computed_blocks_docs(request)
        return [request.document_lens_padded[i] == num_computed_tokens_docs[i] 
                for i in range(len(request.document_lens))]

    def add_doc_request(self, doc_idx: int, request: Request) -> None:
        assert request.has_documents
        logger.debug(f"request {request.request_id} "
                     f"doc {doc_idx} not ready (add to waiting), "
                     f"hash {sha256(tuple(request.documents_token_ids_padded[doc_idx]))}")
        # Spawn a new request for the document and add it to the top of waiting
        sampling_params = copy.deepcopy(request.sampling_params)
        sampling_params.max_tokens = 1 # TODO(haocheng): how to avoid
        req = Request(
            request_id=f"{request.request_id}_d{doc_idx}",
            prompt_token_ids=request.documents_token_ids_padded[doc_idx],
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_positions,
            sampling_params=sampling_params,
            eos_token_id=request.eos_token_id,
            is_document_request=True,
            arrival_time=request.arrival_time,
        )
        self.add_request(req, left=True)

def metadata_for_lazy_attention(request: Request, block_size: int) -> tuple[list[int], list[int]]:
    """Generate the metadata for lazy attention."""
    num_docs = len(request.document_lens)
    num_blocks = len(request.all_token_ids) // block_size
    q_mask = np.zeros(num_blocks + 1, dtype=np.int32)
    q_offset = np.zeros(num_blocks + 1, dtype=np.int32)
    accu_blk = 0
    
    total_padding = sum(request.document_lens_padded) - sum(request.document_lens)
    position_distance = -total_padding
    for doc_idx in range(num_docs):
        num_blks = request.document_lens[doc_idx] // block_size
        # q_offset[accu_blk: accu_blk+num_blks] = position_distance  # No need, since position is processed in Scheduler
        accu_blk += num_blks
        position_distance -= request.document_lens[doc_idx]
        q_mask[accu_blk-1] = request.document_lens_padded[doc_idx] - request.document_lens[doc_idx]
    return list(q_offset), list(q_mask)


original_scheduler = None

def apply_patch():
    import vllm.v1.core.sched.scheduler
    global original_scheduler
    original_scheduler = vllm.v1.core.sched.scheduler.Scheduler
    vllm.v1.core.sched.scheduler.Scheduler = LazyScheduler

def revert_patch():
    import vllm.v1.core.sched.scheduler
    vllm.v1.core.sched.scheduler.Scheduler = original_scheduler
    
    
# ////////////////////////////////////
# Helper functions

def dump_dequeue(
    queue: deque,
    max_num_items: int = 10,
) -> str:
    """Dump the contents of a deque to a string."""
    items = list(queue)
    if len(items) > max_num_items:
        items = items[:max_num_items] + ["..."]
    return str(items)
