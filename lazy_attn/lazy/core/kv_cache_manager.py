"""
A KV cache manager with lazy attention support.

Changed by Haocheng at 2024-09-07
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import cdiv, sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.core.specialized_manager import get_specialized_manager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import RequestStatus

# Import the original manager
from vllm.v1.core.kv_cache_manager import KVCacheManager

# Extra imports for lazy attention
from itertools import chain
from lazy.request import LazyRequest as Request
from lazy.core.kv_cache_utils import hash_request_tokens_with_doc_hash, hash_request_tokens_docs


logger = init_logger(__name__)

class LazyKVCacheManager(KVCacheManager):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # self.req_to_blocks_docs: defaultdict[
        #     str, list[list[KVCacheBlock]]] = defaultdict(list)
        
        self.req_to_block_hashes_docs: defaultdict[
            str, list[list[BlockHashType]]] = defaultdict(list)

        self.num_cached_block_docs: dict[str, list[int]] = {}

        # Cumulative VRAM document-KV cache hit accounting (RQ2 / Table 1).
        # The base get_computed_blocks records prefix_cache_stats, but the lazy
        # per-document reuse path (get_computed_blocks_docs) did not -- so the
        # document-KV hit ratio was never measured. Track it here and log the
        # running ratio so the KV-budget sweep can read it from the job log.
        self.doc_cache_queries = 0
        self.doc_cache_hits = 0
        self._doc_hit_last_log = 0

    def get_computed_blocks(
            self, request: Request) -> tuple[list[KVCacheBlock], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [], 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
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
        # When the request requires prompt logprobs, we skip prefix caching.
        if request.sampling_params.prompt_logprobs is not None:
            return [], 0

        if (len(block_hashes) * self.block_size == request.num_tokens and
            not request.is_document_request):
            # The request is fully cached and need generated tokens,
            # then we not to recompute the last block.
            last_block_hash = block_hashes.pop()
        else:
            last_block_hash = None

        computed_blocks = (
            self.specialized_manager.find_longest_cache_hit(block_hashes))

        if self.use_eagle and len(computed_blocks) > 0:
            # Drop the last matched block if (1) eagle is enabled and
            # (2) there is a cache hit.
            # This is to recompute the last block to get the required
            # hidden states for eagle drafting head.
            computed_blocks.pop()

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)

        if last_block_hash is not None:
            # Add back the last block hash if it was removed.
            # NOTE: Because block_hashes is cached in req_to_block_hashes,
            # we shouldn't modify it directly.
            block_hashes.append(last_block_hash)

        # NOTE(woosuk): Since incomplete blocks are not eligible for
        # sharing, `num_computed_tokens` is always a multiple of
        # `block_size`.
        num_computed_tokens = len(computed_blocks) * self.block_size
        return computed_blocks, num_computed_tokens
        
    def get_computed_blocks_docs(
            self, request: Request,
            drop_first_cached_block: bool = False,
            ) -> tuple[list[list[KVCacheBlock]], list[int]]:
        """Get the computed (cached) blocks for each documents in the request.
        Note that the computed blocks must be full.
        
        Is processing the documents, make sure to retrieve the
           computed blocks for each document independently with prefix caching.
        """
        assert request.has_documents, "Request does not have documents"
        num_docs = len(request.documents_token_ids_padded)
        computed_blocks_docs = [[] for _ in range(num_docs)]
        num_computed_tokens_docs = [0 for _ in range(num_docs)]
        if not self.enable_caching:
            return computed_blocks_docs, num_computed_tokens_docs

        block_hashes_docs = self.req_to_block_hashes_docs[request.request_id]
        
        if not block_hashes_docs:
            block_hashes_docs = hash_request_tokens_docs(self.caching_hash_fn,
                                                         self.block_size, request)
            self.req_to_block_hashes_docs[request.request_id] = block_hashes_docs

        # Then find the computed blocks.
        call_queries = 0
        call_hits = 0
        for doc_idx in range(num_docs):
            block_hashes_doc = block_hashes_docs[doc_idx]
            hit_blocks = (
                self.specialized_manager.find_longest_cache_hit(block_hashes_doc))
            # VRAM hit ratio: count blocks physically in cache BEFORE the
            # lazy first-block recompute trim (those blocks are still cached).
            call_queries += len(block_hashes_doc)
            call_hits += len(hit_blocks)
            computed_blocks_docs[doc_idx] = hit_blocks
            if drop_first_cached_block and computed_blocks_docs[doc_idx]:
                computed_blocks_docs[doc_idx] = computed_blocks_docs[doc_idx][1:]
            num_computed_tokens_docs[doc_idx] = (
                len(computed_blocks_docs[doc_idx]) * self.block_size)
            logger.debug(f"Document {doc_idx} of request {request.request_id} "
                         f"has {num_computed_tokens_docs[doc_idx]} tokens cached.")

        # Record document-KV cache hit stats (RQ2 / Table 1).
        self.doc_cache_queries += call_queries
        self.doc_cache_hits += call_hits
        if self.log_stats and self.prefix_cache_stats is not None:
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += call_queries
            self.prefix_cache_stats.hits += call_hits
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
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        num_lookahead_tokens: int = 0,
    ) -> Optional[list[KVCacheBlock]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []

        req_blocks = self.req_to_blocks[request.request_id]

        if request.is_document_request:
            # Document request no decoding, no need to allocate new blocks.
            num_lookahead_tokens = 0
        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        removed_blocks = self.specialized_manager.remove_skipped_blocks(
            req_blocks, request.num_computed_tokens)
        self.block_pool.free_blocks(removed_blocks)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               len(new_computed_blocks) * self.block_size)
        num_required_blocks = cdiv(
            num_computed_tokens + num_tokens + num_lookahead_tokens,
            self.block_size)
        # logger.info(f"Request {request.request_id} has "
        #              f"{num_computed_tokens} computed tokens, "
        #              f"{num_tokens} new tokens, "
        #              f"{len(new_computed_blocks)} new computed blocks, "
        #              f"{num_lookahead_tokens} lookahead tokens, "
        #              f"and requires {num_required_blocks} blocks.")
        
        num_new_blocks = (num_required_blocks - len(req_blocks) -
                          len(new_computed_blocks))

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(1 for blk in new_computed_blocks
                                            if blk.ref_cnt == 0)
        if (num_new_blocks > self.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not new_computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            block_table_limit = self.max_num_blocks_per_req - len(req_blocks)

            # logger.info(f"!!! {self.block_pool.get_num_free_blocks()} free blocks")
            num_new_blocks = min(
                num_new_blocks,
                self.block_pool.get_num_free_blocks(),
                # Should not exceed the maximum number of blocks per request.
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                block_table_limit,
            )
            assert num_new_blocks > 0

            # Concatenate the computed block IDs and the new block IDs.
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(request.request_id,
                                                      len(new_computed_blocks))
        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        num_full_blocks_after_append = (num_computed_tokens + num_tokens - len(
            request.spec_token_ids)) // self.block_size

        
        # logger.info(f"Request {request.request_id} has "
        #             f"{num_cached_blocks} cached blocks, "
        #             f"{len(new_computed_blocks)} new computed blocks, "
        #             f"and {len(new_blocks)} new blocks allocated, "
        #             f"and will have {num_full_blocks_after_append} full blocks after appending "
        #             f"the new tokens (excluding {len(request.spec_token_ids)} "
        #             f"speculated tokens).")
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
            hash_fn=self.caching_hash_fn,
        )

        self.num_cached_block[
            request.request_id] = num_full_blocks_after_append
        return new_blocks


    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)
        # TODO(haocheng): should we remove immediately?
        self.req_to_block_hashes_docs.pop(request.request_id, None)
