"""
A KV cache manager with lazy attention support.

Changed by Haocheng at 2024-09-07
Ported to vLLM 0.9.2 (KVCacheCoordinator API) at 2026-08-15
"""

from collections import defaultdict
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

# Import the original manager
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import hash_request_tokens

# Extra imports for lazy attention
from lazy.request import LazyRequest as Request
from lazy.core.kv_cache_utils import (hash_request_tokens_with_doc_hash,
                                      hash_request_tokens_docs)


logger = init_logger(__name__)


class LazyKVCacheManager(KVCacheManager):
    """KVCacheManager that can resolve prefix-cache hits per *document*.

    Since vLLM 0.9.0 the manager no longer owns a `specialized_manager`; block
    bookkeeping lives behind `self.coordinator`, which is also what exposes
    `find_longest_cache_hit`. The lazy behaviour is unchanged -- only the calls
    that reach into vLLM internals were rewritten.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.req_to_block_hashes_docs: defaultdict[
            str, list[list[BlockHash]]] = defaultdict(list)

        self.num_cached_block_docs: dict[str, list[int]] = {}

        # Cumulative VRAM document-KV cache hit accounting (RQ2 / Table 1).
        # The base get_computed_blocks records prefix_cache_stats, but the lazy
        # per-document reuse path (get_computed_blocks_docs) did not -- so the
        # document-KV hit ratio was never measured. Track it here and log the
        # running ratio so the KV-budget sweep can read it from the job log.
        self.doc_cache_queries = 0
        self.doc_cache_hits = 0
        self._doc_hit_last_log = 0

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - The blocks that are computed for the request.
                - The number of computed tokens.
        """
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            assert self.block_size is not None
            if request.has_documents:
                # The first block is the document sequence hash.
                block_hashes = hash_request_tokens_with_doc_hash(
                    self.caching_hash_fn, self.block_size, request)
            else:
                block_hashes = hash_request_tokens(self.caching_hash_fn,
                                                   self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1

        # NOTE: When all tokens hit the cache, the base manager must recompute
        # the last token to obtain logits, so it caps the hit length at
        # num_tokens - 1. A *document* request never samples a token -- it only
        # populates the KV cache -- so it may be served entirely from cache.
        #
        # The EAGLE trim (drop the last matched block so the drafting head can
        # recompute its hidden state) is not applied here: since 0.9.0 it lives
        # inside `find_longest_cache_hit`, which reads `use_eagle` off the
        # coordinator we delegate to below.
        if request.is_document_request:
            max_cache_hit_length = request.num_tokens
        else:
            max_cache_hit_length = request.num_tokens - 1

        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens

    def get_computed_blocks_docs(
        self,
        request: Request,
        drop_first_cached_block: bool = False,
    ) -> tuple[list[KVCacheBlocks], list[int]]:
        """Get the computed (cached) blocks for each document in the request.
        Note that the computed blocks must be full.

        When processing the documents, we retrieve the computed blocks for each
        document independently with prefix caching -- this is what lets a
        document be reused regardless of the slot it occupies.
        """
        assert request.has_documents, "Request does not have documents"
        num_docs = len(request.documents_token_ids_padded)
        computed_blocks_docs: list[KVCacheBlocks] = [
            self.create_empty_block_list() for _ in range(num_docs)
        ]
        num_computed_tokens_docs = [0 for _ in range(num_docs)]
        if not self.enable_caching:
            return computed_blocks_docs, num_computed_tokens_docs

        assert self.block_size is not None
        block_hashes_docs = self.req_to_block_hashes_docs[request.request_id]

        if not block_hashes_docs:
            block_hashes_docs = hash_request_tokens_docs(self.caching_hash_fn,
                                                         self.block_size,
                                                         request)
            self.req_to_block_hashes_docs[request.request_id] = block_hashes_docs

        # Then find the computed blocks.
        call_queries = 0
        call_hits = 0
        for doc_idx in range(num_docs):
            block_hashes_doc = block_hashes_docs[doc_idx]
            # A document is position-agnostic and never sampled from, so the
            # full document length is eligible for a cache hit.
            hit_blocks, num_hit_tokens = (
                self.coordinator.find_longest_cache_hit(
                    block_hashes_doc,
                    len(block_hashes_doc) * self.block_size))
            # VRAM hit ratio: count blocks physically in cache BEFORE the
            # lazy first-block recompute trim (those blocks are still cached).
            call_queries += len(block_hashes_doc)
            call_hits += num_hit_tokens // self.block_size

            if drop_first_cached_block and any(hit_blocks):
                # Drop the first block of every group so it is recomputed.
                hit_blocks = tuple(group[1:] for group in hit_blocks)
                num_hit_tokens = max(num_hit_tokens - self.block_size, 0)

            computed_blocks_docs[doc_idx] = KVCacheBlocks(hit_blocks)
            num_computed_tokens_docs[doc_idx] = num_hit_tokens
            logger.debug(f"Document {doc_idx} of request {request.request_id} "
                         f"has {num_computed_tokens_docs[doc_idx]} tokens cached.")

        # Record document-KV cache hit stats (RQ2 / Table 1).
        self.doc_cache_queries += call_queries
        self.doc_cache_hits += call_hits
        if self.log_stats and self.prefix_cache_stats is not None:
            # PrefixCacheStats counts *tokens* -- get_computed_blocks adds
            # request.num_tokens / num_new_computed_tokens -- so convert the
            # per-document block counts before mixing them into the same
            # aggregate, or every document contribution is understated by a
            # factor of block_size.
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += call_queries * self.block_size
            self.prefix_cache_stats.hits += call_hits * self.block_size
        if self.doc_cache_queries - self._doc_hit_last_log >= 200:
            self._doc_hit_last_log = self.doc_cache_queries
            ratio = self.doc_cache_hits / max(self.doc_cache_queries, 1)
            logger.info(
                f"[LAZY_DOC_KV_HIT] hits={self.doc_cache_hits} "
                f"queries={self.doc_cache_queries} ratio={ratio:.4f}")
        return computed_blocks_docs, num_computed_tokens_docs

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_draft_tokens: int = 0,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Identical to the base implementation except that a document request is
        never decoded from, so it must not reserve lookahead slots.
        """
        if request.is_document_request:
            # Document request no decoding, no need to allocate new blocks.
            num_lookahead_tokens = 0

        return super().allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_draft_tokens=num_draft_tokens,
            num_lookahead_tokens=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
        )

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)
        # TODO(haocheng): should we remove immediately?
        self.req_to_block_hashes_docs.pop(request.request_id, None)
