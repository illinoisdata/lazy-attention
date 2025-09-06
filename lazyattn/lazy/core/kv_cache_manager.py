from itertools import chain
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional, Union

from vllm.logger import init_logger
from vllm.utils import cdiv, sha256
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.core.specialized_manager import get_specialized_manager
from vllm.v1.kv_cache_interface import KVCacheConfig
# from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import RequestStatus
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.block_pool import BlockPool

from lazy.request import LazyRequest as Request
from lazy.metrics.stats import PrefixCacheStats

from lazy.core.kv_cache_utils import hash_request_tokens_with_doc_hash, hash_request_tokens_docs
from lazy.core.block_pool import cache_full_blocks, cache_full_blocks_docs

class LazyKVCacheManager(KVCacheManager):

    def __init__(*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.req_to_blocks_docs: defaultdict[
            str, list[list[KVCacheBlock]]] = defaultdict(list)
        
        self.req_to_block_hashes_docs: defaultdict[
            str, list[list[BlockHashType]]] = defaultdict(list)
        
        self.num_cached_block_docs: dict[str, list[int]] = {}

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
            self.prefix_cache_stats.requests += 1
        if request.sampling_params.prompt_logprobs is None:
            if len(block_hashes) * self.block_size == request.num_tokens and \
                'd' not in request.request_id:
                # When prompt length is divisible by the block size and all
                # blocks are cached, we need to recompute the last token. This
                # have to be achieved by re-computing an entire block because
                # allocate_slots() assumes num_computed_tokens is always a
                # multiple of the block size. To achieve this, remove the last
                # block hash from the block_hashes for find_longest_cache_hit
                # This limitation can potentially be removed in the future to
                # slightly improve the performance.
                last_block_hash = block_hashes.pop()
            else:
                last_block_hash = None

            computed_blocks = (
                self.specialized_manager.find_longest_cache_hit(block_hashes))

            if last_block_hash is not None:
                # Add back the last block hash if it was removed.
                block_hashes.append(last_block_hash)

            if self.log_stats:
                self.prefix_cache_stats.queries += len(block_hashes)
                self.prefix_cache_stats.hits += len(computed_blocks)

            # NOTE(woosuk): Since incomplete blocks are not eligible for
            # sharing, `num_computed_tokens` is always a multiple of
            # `block_size`.
            num_computed_tokens = len(computed_blocks) * self.block_size
            return computed_blocks, num_computed_tokens
        else:
            # Skip cache hits for prompt logprobs
            return [], 0
        
    def get_computed_blocks_docs(
            self, request: Request, 
            ) -> tuple[list[list[KVCacheBlock]], list[int]]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.
        
        Is processing the documents, make sure to retrieve the
           computed blocks for the documents indenpently with prefix caching.
        """
        assert request.has_documents, "Request does not have documents"
        num_docs = len(request.documents_token_ids_padded)
        if not self.enable_caching:
            return [[] for _ in range(num_docs)], \
                    [0 for _ in range(num_docs)]

        block_hashes_docs = self.req_to_block_hashes_docs[request.request_id]
        
        if not block_hashes_docs:
            block_hashes_docs = hash_request_tokens_docs(self.caching_hash_fn,
                                                         self.block_size, request)
            self.req_to_block_hashes_docs[request.request_id] = block_hashes_docs

        # if self.log_stats:
        #     self.prefix_cache_stats.doc_requests += 1
        # Then find the computed blocks.
        computed_blocks_docs = [[] for _ in range(num_docs)]
        num_computed_tokens_docs = [0 for _ in range(num_docs)]
        for doc_idx in range(num_docs):
            block_hashes_doc = block_hashes_docs[doc_idx]
            computed_blocks_docs[doc_idx] = (
                self.specialized_manager.find_longest_cache_hit(block_hashes_doc))
            num_computed_tokens_docs[doc_idx] = (
                len(computed_blocks_docs[doc_idx]) * self.block_size)
            
        # Update stats information TODO(haocheng): utilize the stats
        # if self.log_stats:
        #     self.prefix_cache_stats.doc_hits += num_docs
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
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.

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

        if 'd' in request.request_id:
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
        num_required_blocks = cdiv(num_computed_tokens + num_tokens + num_lookahead_tokens,
                                   self.block_size)
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
            if request.has_documents:
                block_table_limit -= sum([len(req_blocks_doc) for req_blocks_doc in
                                          self.req_to_blocks_docs[request.request_id]])

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

        # print(f"*** A = {len(self.req_to_block_hashes[request.request_id])}")
        # print(f"*** B = {num_cached_blocks}")
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

    def allocate_slots_docs(
        self,
        request: Request,
        num_tokens_docs: Optional[list[int]] = None,
        new_computed_blocks_docs: Optional[list[list[KVCacheBlock]]] = None,
    ) -> Optional[list[list[KVCacheBlock]]]:
        if sum(num_tokens_docs) == 0:
            raise ValueError("sum of num_tokens_docs must be greater than 0")
        
        num_docs = len(request.documents_token_ids_padded)
        new_computed_blocks_docs = new_computed_blocks_docs or [[] for _ in range(num_docs)]
        req_blocks_docs = self.req_to_blocks_docs[request.request_id]

        # TODO(haocheng): we can free some blocks in the future

        num_computed_tokens_docs = [
            request.num_computed_tokens_docs[doc_idx] +
            len(new_computed_blocks_docs[doc_idx]) * self.block_size
            for doc_idx in range(num_docs)
        ]
        num_required_blocks_docs = [
            cdiv(num_computed_tokens_docs[doc_idx] + num_tokens_docs[doc_idx],
                self.block_size) for doc_idx in range(num_docs)
        ]
        num_new_blocks_docs = [
                (num_required_blocks_docs[doc_idx] - 
                 len(req_blocks_docs[doc_idx]) -
                 len(new_computed_blocks_docs[doc_idx]))
                for doc_idx in range(num_docs)
        ]
        num_evictable_computed_blocks_docs = sum(1 for blk in chain.from_iterable(
                new_computed_blocks_docs) if blk.ref_cnt == 0)
        if (sum(num_new_blocks_docs) > self.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks_docs):
                # Cannot allocate new blocks
            return None
        # Touch the add ref count for the computed blocks
        assert self.enable_caching, "Should not be here if caching is disabled"
        self.block_pool.touch(chain.from_iterable(new_computed_blocks_docs))
        
        # Append the new computed blocks to the request blocks until now to
        for doc_idx in range(num_docs):
            req_blocks_docs[doc_idx].extend(new_computed_blocks_docs[doc_idx])

        if sum(num_new_blocks_docs) <= 0:
            # No new block is needed.
            new_blocks_docs = [[] for _ in range(num_docs)]
        else:
            total_num_req_blocks_docs = sum(
                len(req_blocks_docs[doc_idx]) 
                for doc_idx in range(num_docs)
            )
            new_blocks_docs = [[] for _ in range(num_docs)]
            for doc_idx in range(num_docs):
                # We do not give preallocated blocks for documents
                num_new_blocks_doc = min(
                    num_new_blocks_docs[doc_idx],
                    self.block_pool.get_num_free_blocks(),
                )
                if num_new_blocks_doc + total_num_req_blocks_docs > \
                        self.max_num_blocks_per_req:
                    raise ValueError("Exceed the maximum number of blocks per request")
                total_num_req_blocks_docs += num_new_blocks_doc
                
                assert num_new_blocks_doc > 0
                
                new_blocks_doc = self.block_pool.get_new_blocks(num_new_blocks_docs[doc_idx])
                new_blocks_docs[doc_idx].extend(new_blocks_doc)
                req_blocks_docs[doc_idx].extend(new_blocks_doc)
                
            if not self.enable_caching:
                return new_blocks_docs
            
            num_cached_blocks_docs = self.num_cached_block_docs.get(
                request.request_id,
                [
                    len(new_computed_blocks_doc)
                    for new_computed_blocks_doc in new_computed_blocks_docs
                ]
            )
            
            num_full_blocks_after_append_docs = [
                (num_computed_tokens_docs[doc_idx] +
                 num_tokens_docs[doc_idx]) // self.block_size
                for doc_idx in range(num_docs)]

        self.block_pool.cache_full_blocks_docs(
            request=request,
            block_docs=req_blocks_docs,
            block_hashes_docs=self.req_to_block_hashes_docs[request.request_id],
            num_cached_blocks_docs=num_cached_blocks_docs,
            num_full_blocks_docs=num_full_blocks_after_append_docs,
            block_size=self.block_size,
            hash_fn=self.caching_hash_fn,
        )
            
        self.num_cached_block_docs[
            request.request_id] = num_full_blocks_after_append_docs
        return new_blocks_docs
    
    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, [])
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request.request_id, None)

    # TODO(haocheng): consider resetting the prefix cache
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if self.block_pool.reset_prefix_cache():
            self.prefix_cache_stats.reset = True
            return True
        return False

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            int: The number of common prefix blocks.
        """
        assert request.status == RequestStatus.RUNNING
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks
    
    # TODO(haocheng): add a function `get_num_common_doc_blocks`

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)
        # TODO(haocheng): should we remove immediately?
        self.req_to_block_hashes_docs.pop(request.request_id, None)
        
        
    def print_stats(self):
        if self.log_stats:
            with open("prefix_cache_stats.txt", "w") as f:
                f.write(str(self.prefix_cache_stats.memory_footprint))
        else:
            with open("prefix_cache_stats.txt", "w") as f:
                f.write("Prefix cache stats are not logged.")
