import time
from typing import List, Tuple, Dict, Optional, Set
from torch import Tensor
from drag.RAGRequest import RAGRequest, CacheDocRequest, QueryRequest, RAGRequestType

class KV_Cache_Manager:
    enable_swap_in_cpu_blocks = False
    def __init__(self, block_size: int, gpu_kv_cache: List[Tensor], cpu_kv_cache: List[Tensor], num_gpu_blocks: int, num_cpu_blocks: int):
        pass
    def free(request: RAGRequest) -> None:
        pass

    def append_slots(request: RAGRequest, num_new_tokens: int) -> Tuple[List[int], List[int]]:
        slot_mapping = []
        new_block_ids = []
        pass

    def __init__(self, block_size: int, gpu_kv_cache: List[Tensor], cpu_kv_cache: List[Tensor],
                 num_gpu_blocks: int, num_cpu_blocks: int):
        self.block_size = block_size
        self.gpu_kv_cache = gpu_kv_cache
        self.cpu_kv_cache = cpu_kv_cache

        self._free_gpu_blocks: List[int] = list(range(num_gpu_blocks))
        self._free_cpu_blocks: List[int] = list(range(num_cpu_blocks))

        self._total_gpu_blocks = num_gpu_blocks
        self._total_cpu_blocks = num_cpu_blocks

        #tracking data structures in maanger
        #   _request_blocks: maps request_id to allocated GPU block IDs.
        #   _request_slots: maps request_id to the slot mapping for tokens.
        #   _cpu_blocks: maps request_id to allocated CPU block IDs (for swapped-out data).
        self._request_blocks: Dict[int, List[int]] = {}
        self._request_slots: Dict[int, List[int]] = {}
        self._cpu_blocks: Dict[int, List[int]] = {}

        # data structures for eviction
        # _block_info: maps block_id to its usage info (ref_count and last_used timestamp).
        self._block_info: Dict[int, Dict[str, float]] = {}

        # for CacheDocRequests, tracking which document is cached.
        self._doc_to_blocks: Dict[int, List[int]] = {} 
        self._doc_to_req: Dict[int, int] = {} # maintains mapping of a doc being used for which requests {doc_id: [req_id,..]}

        # running requests i.e. docs related to running reqs that must not be evicted
        self._running_requests: Set[int] = set()

    def update_running_requests(self, running_req_ids: List[int]) -> None:
        """
        Update the set of request IDs that are currently running and whose blocks
        should not be evicted.
        """
        self._running_requests = set(running_req_ids)

    def free(self, request: RAGRequest) -> None:
        """Release all allocated blocks (GPU and CPU) for the given request."""
        request_id = request.request_id
        if request_id not in self._request_blocks:
            return  # already freed or not registered

        self._free_gpu_blocks.extend(self._request_blocks[request_id])
        for block in self._request_blocks[request_id]:
            if block in self._block_info:
                del self._block_info[block]
        del self._request_blocks[request_id]

        if request_id in self._cpu_blocks:
            self._free_cpu_blocks.extend(self._cpu_blocks[request_id])
            del self._cpu_blocks[request_id]

        if request_id in self._request_slots:
            del self._request_slots[request_id]

        # if the request is caching a document, remove its entry from _doc_to_blocks.
        if request.get_type() == RAGRequestType.CACHE_DOC:
            cache_req = request  # type: CacheDocRequest
            if cache_req.doc_id in self._doc_to_blocks:
                del self._doc_to_blocks[cache_req.doc_id]
            if cache_req.doc_id in self._doc_to_req:
                del self._doc_to_req[cache_req.doc_id]

    def append_slots(self, request: RAGRequest, num_new_tokens: int) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        Reserve additional GPU blocks (and create a slot mapping) for new tokens.
        Typically used during the decode phase.
        
        Returns a tuple (slot_mapping, new_block_ids). If not enough free GPU blocks are available,
        returns (None, None) to signal failure.
        """
        if num_new_tokens <= 0:
            return ([], [])

        request_id = request.request_id
        needed_blocks = (num_new_tokens + self.block_size - 1) // self.block_size

        if needed_blocks > len(self._free_gpu_blocks):
            return (None, None)  # not enough free GPU blocks

        self._request_blocks.setdefault(request_id, [])
        self._request_slots.setdefault(request_id, [])

        # allocate new blocks
        new_blocks = self._free_gpu_blocks[:needed_blocks]
        self._free_gpu_blocks = self._free_gpu_blocks[needed_blocks:]
        self._request_blocks[request_id].extend(new_blocks)

        now = time.time()
        for block in new_blocks:
            self._block_info[block] = {"ref_count": 1, "last_used": now}

        slot_mapping: List[int] = []
        tokens_assigned = 0
        for block_id in new_blocks:
            tokens_in_block = min(self.block_size, num_new_tokens - tokens_assigned)
            for i in range(tokens_in_block):
                slot_mapping.append(block_id * self.block_size + i)
                tokens_assigned += 1
            if tokens_assigned >= num_new_tokens:
                break

        self._request_slots[request_id].extend(slot_mapping)
        return (slot_mapping, new_blocks)

    def allocate_slots(self, request: RAGRequest, num_new_tokens: int) -> Tuple[List[int], List[int]]:
        """
        Allocate GPU blocks and build a slot mapping for the requested tokens.
        When num_new_tokens is zero, this acts as a registration call.
        
        For CacheDocRequests, if not enough free GPU blocks exist,
        the evictor is triggered to free up space.
        
        Returns a tuple (slot_mapping, new_block_ids).
        """
        request_id = request.request_id
        slot_mapping: List[int] = []
        new_block_ids: List[int] = []

        self._request_blocks.setdefault(request_id, [])
        self._request_slots.setdefault(request_id, [])

        needed_blocks = (num_new_tokens + self.block_size - 1) // self.block_size

        # for document caching, if insufficient free blocks exist, we do eviction
        if request.get_type() == RAGRequestType.CACHE_DOC and needed_blocks > len(self._free_gpu_blocks):
            self._evict_blocks(needed_blocks - len(self._free_gpu_blocks))

        if needed_blocks > len(self._free_gpu_blocks):
            raise RuntimeError(f"Not enough free blocks even after eviction. Needed {needed_blocks}, have {len(self._free_gpu_blocks)}")

        new_block_ids = self._free_gpu_blocks[:needed_blocks]
        self._free_gpu_blocks = self._free_gpu_blocks[needed_blocks:]
        self._request_blocks[request_id].extend(new_block_ids)

        now = time.time()
        for block in new_block_ids:
            self._block_info[block] = {"ref_count": 1, "last_used": now}

        tokens_assigned = 0
        for block_id in new_block_ids:
            tokens_in_block = min(self.block_size, num_new_tokens - tokens_assigned)
            for i in range(tokens_in_block):
                slot_mapping.append(block_id * self.block_size + i)
                tokens_assigned += 1
            if tokens_assigned >= num_new_tokens:
                break

        self._request_slots[request_id].extend(slot_mapping)

        # for CacheDocRequests, record in the doc mapping
        if request.get_type() == RAGRequestType.CACHE_DOC:
            cache_req = request  # type: CacheDocRequest
            self._doc_to_blocks[cache_req.doc_id] = self._request_blocks[request_id].copy()
            self._doc_to_req[cache_req.doc_id] = request_id

        return (slot_mapping, new_block_ids)

    def get_computed_gpu_blocks(self, request: RAGRequest, swap_in_cpu_blocks: bool = False) -> List[int]:
        """
        Returns the list of GPU block IDs that have already been computed (cached) for the given request.
        For CacheDocRequests, if the document is cached (tracked in _doc_to_blocks), that copy is returned.
        For other requests, internal bookkeeping is used.
        If swap_in_cpu_blocks is True and swapping is enabled, attempts to swap CPU blocks into GPU memory.
        
        In either case, each returned block has its last_used timestamp updated and its ref_count incremented.
        """
        now = time.time()
        # for CacheDocRequests, check our internal doc mapping first
        if request.get_type() == RAGRequestType.CACHE_DOC:
            cache_req = request  # type: CacheDocRequest
            if cache_req.doc_id in self._doc_to_blocks:
                blocks = self._doc_to_blocks[cache_req.doc_id].copy()
                for block in blocks:
                    if block in self._block_info:
                        self._block_info[block]["last_used"] = now
                        self._block_info[block]["ref_count"] += 1
                return blocks

        if request.request_id not in self._request_blocks:
            return []
        gpu_blocks = self._request_blocks[request.request_id].copy()
        for block in gpu_blocks:
            if block in self._block_info:
                self._block_info[block]["last_used"] = now
                self._block_info[block]["ref_count"] += 1

        # if swapping is enabled
        if swap_in_cpu_blocks and self.enable_swap_in_cpu_blocks and request.request_id in self._cpu_blocks:
            cpu_blocks = self._cpu_blocks[request.request_id]
            if len(cpu_blocks) > len(self._free_gpu_blocks):
                raise RuntimeError(f"Not enough free GPU blocks to swap in. Need {len(cpu_blocks)}, have {len(self._free_gpu_blocks)}")
            new_gpu_blocks = self._free_gpu_blocks[:len(cpu_blocks)]
            self._free_gpu_blocks = self._free_gpu_blocks[len(cpu_blocks):]
            self._request_blocks[request.request_id].extend(new_gpu_blocks)
            gpu_blocks.extend(new_gpu_blocks)
            self._free_cpu_blocks.extend(cpu_blocks)
            del self._cpu_blocks[request.request_id]
            for block in new_gpu_blocks:
                if block in self._block_info:
                    self._block_info[block]["last_used"] = now
                    self._block_info[block]["ref_count"] += 1
        return gpu_blocks

    def num_free_blocks(self) -> int:
        """Return the number of free GPU blocks."""
        return len(self._free_gpu_blocks)

    def _evict_blocks(self, num_required: int) -> None:
        """
        Evict blocks from cached documents until at least num_required free GPU blocks are available.
        The eviction policy selects documents based on the lowest average reference count and oldest usage,
        but only considers candidates whose associated request is not in the protected (running) set.
        """
        while len(self._free_gpu_blocks) < num_required and self._doc_to_blocks:
            candidate_doc = None
            candidate_score = None
            for doc_id, block_list in self._doc_to_blocks.items():
                req_id = self._doc_to_req.get(doc_id)
                # skip eviction if the document's request is currently running
                if req_id in self._running_requests:
                    continue
                total_ref = 0
                min_time = float('inf')
                count = 0
                for block in block_list:
                    info = self._block_info.get(block, {"ref_count": 1, "last_used": 0})
                    total_ref += info["ref_count"]
                    min_time = min(min_time, info["last_used"])
                    count += 1
                avg_ref = total_ref / count if count > 0 else 0
                score = (avg_ref, min_time)  # lower average ref and older blocks are prioritized
                if candidate_score is None or score < candidate_score:
                    candidate_score = score
                    candidate_doc = doc_id
            if candidate_doc is None:
                break  # no eviction
            self._evict_doc(candidate_doc)

    def _evict_doc(self, doc_id: int) -> None:
        """
        Evict the cached blocks associated with the given doc_id.
        The blocks are removed from the corresponding request and returned to the free pool.
        """
        req_id = self._doc_to_req.get(doc_id)
        if req_id is None:
            return
        blocks = self._doc_to_blocks.get(doc_id, [])
        # remove these blocks from the request's allocation
        if req_id in self._request_blocks:
            for block in blocks:
                if block in self._request_blocks[req_id]:
                    self._request_blocks[req_id].remove(block)
                if block in self._block_info:
                    del self._block_info[block]
        # return blocks to free pool
        self._free_gpu_blocks.extend(blocks)
        # remove eviction mappings
        if doc_id in self._doc_to_blocks:
            del self._doc_to_blocks[doc_id]
        if doc_id in self._doc_to_req:
            del self._doc_to_req[doc_id]
