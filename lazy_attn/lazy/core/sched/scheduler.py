# SPDX-License-Identifier: Apache-2.0

"""
Core of LazyAttention

Changed by Haocheng at 2025/09/04
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
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
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
import numpy as np
from vllm.utils import cdiv, sha256

# Overwrite classes
from lazy.core.kv_cache_manager import LazyKVCacheManager
from lazy.request import RequestStatus
from lazy.request import LazyRequest as Request
from lazy.engine import EngineCoreRequest, EngineCoreEventType
from lazy.core.sched.output import NewRequestData
from lazy.utils.variants import (get_lazy_attention_variant_code,
                                 is_mepic_variant,
                                 lazy_shared_kv_profile_enabled,
                                 lazy_shared_kv_profile_min_reqs,
                                 mepic_first_block_recompute_enabled)

# For patch
from vllm.v1.core.sched.scheduler import Scheduler

# NOTE(Haocheng)
# Different from the original V1Scheduler, this scheduler need to process the 
# documents inner the request
# Here we abort the constraint of num running request, since our 
# request can have multiple subrequests, and we need to process all of them
# in the same time.

LAZY_ATTENTION_VARIANT = get_lazy_attention_variant_code()
IS_MEPIC = is_mepic_variant()
MEPIC_FIRST_BLOCK_RECOMPUTE = mepic_first_block_recompute_enabled()
LAZY_SHARED_KV_PROFILE = lazy_shared_kv_profile_enabled()
LAZY_SHARED_KV_PROFILE_MIN_REQS = lazy_shared_kv_profile_min_reqs()


class LazyScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        logger.info("Initializing LazyScheduler")
        # Delegate to the upstream Scheduler rather than re-copying its
        # __init__. The only lazy-specific piece of setup is swapping in a
        # KV cache manager that can resolve prefix hits per document, so
        # everything else stays in sync with vLLM automatically.
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        self.block_size = self.cache_config.block_size

        self.kv_cache_manager = LazyKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )

        logger.info("LazyScheduler launched")

    def schedule(self) -> SchedulerOutput:
        schedule_start = time.perf_counter() if LAZY_SHARED_KV_PROFILE else 0.0
        logger.debug(f"Scheduler: waiting={dump_dequeue(self.waiting)}, "
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

        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
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
        lazy_metadata_requests = 0
        lazy_metadata_blocks = 0
        lazy_doc_merges = 0

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

            # Tokens scheduled past the request's own length are speculative
            # drafts; the cache manager must know not to treat them as
            # committed when it caches blocks.
            num_draft_tokens = max(
                num_new_tokens + request.num_computed_tokens -
                request.num_tokens, 0)

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_draft_tokens=num_draft_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
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

                    self.waiting.prepend_request(preempted_req)
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
            req_to_new_block_ids[request.request_id] = (
                new_blocks.get_block_ids())
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

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # ///////////////////////////////////////////////////////////////////////
        # Next, schedule the WAITING requests.
        # NOTE(Haocheng): Here we need to check and assemble query requests and 
        # documents.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
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

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # NOTE(Haocheng): if one request reach here, there are three cases,
                # 1. it does not have documents
                #   1.1. it is a document request
                #   1.2. it is a normal request without documents (no extra process)
                # 2. its documents are all ready at *this* minor step

                # TODO(Haocheng): test whether this can bring benefit
                # fastening Case 1.1
                if request.is_document_request and \
                    num_computed_tokens == request.num_tokens:
                    self.waiting.pop_request()
                    continue

                # NOTE(haocheng): new block is allocated, then we assemble
                # lazy request if needed
                # The role:
                # - General new request attends all documents
                # - Lock all documents by increasing ref cnt
                if request.has_documents:
                    # Case 2 -> Case 1.2
                    request.merge_documents()
                    lazy_doc_merges += 1
                    logger.debug(f"Request {request.request_id} merges "
                                 f"documents, total prompt len "
                                 f"{request.num_prompt_tokens}")

                    drop_first_cached_block = (
                        IS_MEPIC and MEPIC_FIRST_BLOCK_RECOMPUTE
                    )
                    computed_blocks_docs, num_computed_tokens_docs = \
                        self.kv_cache_manager.get_computed_blocks_docs(
                            request,
                            drop_first_cached_block=drop_first_cached_block,
                        )
                    # The documents sit in front of the query in the merged
                    # prompt, so their cached blocks are prepended (in document
                    # order) to the query's own hit. KVCacheBlocks.__add__
                    # concatenates group-wise, which is what the multi-group
                    # block layout in vLLM 0.9.x expects.
                    for doc_blocks in reversed(computed_blocks_docs):
                        new_computed_blocks = doc_blocks + new_computed_blocks
                    expected_doc_tokens = sum(request.document_lens_padded)
                    if not drop_first_cached_block:
                        assert sum(num_computed_tokens_docs) == expected_doc_tokens, (
                            "Cached document lengths do not match")
                    else:
                        logger.debug(
                            "MEPIC first-block recompute enabled for request %s: "
                            "reusing %d/%d cached document tokens.",
                            request.request_id,
                            sum(num_computed_tokens_docs),
                            expected_doc_tokens,
                        )
                    num_computed_tokens += sum(num_computed_tokens_docs)
                    # allocate_slots() is told how many of the computed tokens
                    # are *newly* hit locally, so the document hits must be
                    # counted there too.
                    num_new_local_computed_tokens += sum(
                        num_computed_tokens_docs)
                    logger.debug(f"After merging documents, "
                                 f"request {request.request_id} has "
                                 f"{num_computed_tokens} computed tokens.")

                    # Get metadata for lazy attention
                    (req_to_q_offset[request.request_id], 
                     req_to_q_mask[request.request_id]) = \
                        metadata_for_variant(request, self.block_size)
                    lazy_metadata_requests += 1
                    lazy_metadata_blocks += len(
                        req_to_q_offset[request.request_id])
                    logger.debug(f"Request {request.request_id} has "
                                 f"query offset {req_to_q_offset[request.request_id]} "
                                 f"and query mask {req_to_q_mask[request.request_id]}")

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
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,  # will be touched in allocate_slots
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVConnector: update internal state after allocation.
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                self.waiting.pop_request()
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
                req_to_new_block_ids[request.request_id] = (
                    self.kv_cache_manager.get_block_ids(request.request_id))
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
            self.waiting.prepend_requests(skipped_waiting_requests)

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
            NewRequestData.from_request(
                req,
                req_to_new_block_ids[req.request_id],
                lazy_variant=(LAZY_ATTENTION_VARIANT if req.has_documents else 0),
                q_offset=req_to_q_offset.get(req.request_id, None),
                q_mask=req_to_q_mask.get(req.request_id, None),
            )
            for req in scheduled_new_reqs
        ]
        # Since 0.9.x the cached requests are sent as a single batched
        # CachedRequestData rather than one object per request; reuse the base
        # helper so the batching stays in sync with vLLM.
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
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
        if LAZY_SHARED_KV_PROFILE:
            active_reqs = len(self.running) + len(self.waiting)
            if active_reqs >= LAZY_SHARED_KV_PROFILE_MIN_REQS:
                logger.info(
                    "LazyProfile schedule: active=%d running=%d waiting=%d "
                    "scheduled=%d tokens=%d docs_merged=%d lazy_meta_reqs=%d "
                    "lazy_meta_blocks=%d elapsed_ms=%.3f",
                    active_reqs,
                    len(self.running),
                    len(self.waiting),
                    len(num_scheduled_tokens),
                    total_num_scheduled_tokens,
                    lazy_doc_merges,
                    lazy_metadata_requests,
                    lazy_metadata_blocks,
                    (time.perf_counter() - schedule_start) * 1000.0,
                )
        logger.debug(f"Scheduler output: {scheduler_output}")
        return scheduler_output

    def add_request(self, request: Request, left=False) -> None:
        # NOTE(Haocheng): this function is used to add a request to the waiting
        # queue. For lazy attention, we add the request with `DOC_WAITING`
        tag = "[Normal]"
        if request.is_document_request:
            tag = "[Document]"
        elif request.has_documents:
            tag = "[Lazy]"
        logger.debug(f"Adding {tag} request {request.request_id} to LazyScheduler")
        if left:
            self.waiting.prepend_request(request)
        else:
            self.waiting.add_request(request)
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
        # NOTE(haocheng): new spawned doc has higher priority than other
        # waiting reqs, since it is blocking the query request.
        self.add_request(request.document_request(doc_idx), left=True)

def metadata_for_lazy_attention(request: Request, block_size: int) -> tuple[list[int], list[int]]:
    """Generate the metadata for lazy attention."""
    num_docs = len(request.document_lens)
    # Number of blocks for docs + 1 (for query)
    num_blocks = sum(request.document_lens_padded) // block_size + 1
    q_mask = np.zeros(num_blocks, dtype=np.int32)
    q_offset = np.zeros(num_blocks, dtype=np.int32)
    cursor = 0
    padding_lens = np.array(request.document_lens_padded) - \
                   np.array(request.document_lens)
    # Encode absolute rotation positions with a +1 bias so:
    # - value 0 remains available as a sentinel
    # - value 1 explicitly resets back to the original Q orientation
    abs_rot_pos = int(sum(padding_lens))
    for doc_idx in range(num_docs):
        num_blk_doc = request.document_lens_padded[doc_idx] // block_size
        q_offset[cursor:cursor + num_blk_doc] = abs_rot_pos + 1
        cursor += num_blk_doc
        q_mask[cursor - 1] = padding_lens[doc_idx]
        abs_rot_pos += request.document_lens[doc_idx]

    # Query / decode block should use the original global query RoPE.
    q_offset[cursor] = 1
    return list(q_offset), list(q_mask)


def metadata_for_mepic(request: Request,
                       block_size: int) -> tuple[list[int], list[int]]:
    q_offset, q_mask = metadata_for_lazy_attention(request, block_size)
    return [0 for _ in q_offset], q_mask


def metadata_for_variant(request: Request,
                         block_size: int) -> tuple[list[int], list[int]]:
    if IS_MEPIC:
        return metadata_for_mepic(request, block_size)
    return metadata_for_lazy_attention(request, block_size)

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
