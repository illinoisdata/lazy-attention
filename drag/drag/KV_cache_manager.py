import time
from typing import List, Tuple, Dict, Optional, Set
from torch import Tensor
from drag.RAGRequest import RAGRequest, CacheDocRequest, QueryRequest, RAGRequestType

class KV_Cache_Manager:
    enable_swap_in_cpu_blocks = False
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

    def free(self, request: RAGRequest) -> None:
        """
        Mark the request as finished by decrementing the reference count of each block
        allocated to this request. Blocks are  not immediately returned to the free pool;
        they will be reclaimed later via the eviction process.
        """
        request_id = request.request_id
        if request_id not in self._request_blocks:
            return  # already freed or not registered

        # decrement ref_count for each allocated block
        for block in self._request_blocks[request_id]:
            if block in self._block_info:
                # Decrement the ref count but never below zero
                self._block_info[block]["ref_count"] = max(0, self._block_info[block]["ref_count"] - 1)

        for block in self._request_blocks[request_id]:
            if block in self._block_info and self._block_info[block]["ref_count"] == 0:
                self._free_gpu_blocks.append(block)

        del self._request_blocks[request_id]
        if request_id in self._request_slots:
            del self._request_slots[request_id]
        if request_id in self._cpu_blocks:
            del self._cpu_blocks[request_id]

    def append_slots(self, request: RAGRequest, num_new_tokens: int) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        Reserve additional GPU slots for new tokens, reusing any existing allocated capacity first.
        This is typically used during the decode phase.

        If some blocks have already been allocated but not fully used, the method will
        assign tokens to the remaining capacity. If that capacity is insufficient, it will allocate
        additional blocks from the free pool.

        Returns:
            A tuple (slot_mapping, new_block_ids). If not enough free GPU blocks are available,
            returns (None, None) to signal failure. If no new blocks are allocated, new_block_ids will be [].
        """
        if num_new_tokens <= 0:
            return ([], [])

        request_id = request.request_id
    
        self._request_blocks.setdefault(request_id, [])
        self._request_slots.setdefault(request_id, [])
        current_blocks = self._request_blocks[request_id]
        current_slots = self._request_slots[request_id]

        # check avail slots 
        capacity = len(current_blocks) * self.block_size  # total slots available
        used = len(current_slots)                          # slots already assigned
        available = capacity - used                        # free slots in already allocated blocks

        slot_mapping: List[int] = []
        new_block_ids: List[int] = []

        # case 1: sufficient free capacity in already allocated blocks
        if available >= num_new_tokens:
            for i in range(num_new_tokens):
                global_index = used + i  # next free slot index
                block_index = global_index // self.block_size
                offset = global_index % self.block_size
                block_id = current_blocks[block_index]
                slot_mapping.append(block_id * self.block_size + offset)
            current_slots.extend(slot_mapping)
            return (slot_mapping, [])

        # case 2: use up available capacity first
        if available > 0:
            for i in range(available):
                global_index = used + i
                block_index = global_index // self.block_size
                offset = global_index % self.block_size
                block_id = current_blocks[block_index]
                slot_mapping.append(block_id * self.block_size + offset)
        remaining_tokens = num_new_tokens - available

        # check how many new blocks are required for the remaining tokens
        needed_new_blocks = (remaining_tokens + self.block_size - 1) // self.block_size

        if needed_new_blocks > len(self._free_gpu_blocks):
            self._evict_blocks(needed_new_blocks - len(self._free_gpu_blocks))

        if needed_new_blocks > len(self._free_gpu_blocks):
            raise RuntimeError(f"Not enough free blocks even after eviction. Needed {needed_new_blocks}, have {len(self._free_gpu_blocks)}")

        # allocate new blocks
        new_blocks = self._free_gpu_blocks[:needed_new_blocks]
        self._free_gpu_blocks = self._free_gpu_blocks[needed_new_blocks:]
        self._request_blocks[request_id].extend(new_blocks)
        new_block_ids = new_blocks  # these are the new blocks allocated

        now = time.time()
        for block in new_blocks:
            self._block_info[block] = {"ref_count": 1, "last_used": now}

        # new total capacity after allocation
        old_capacity = capacity  # capacity before new blocks were allocated
        # new slots start from the old capacity index
        for i in range(remaining_tokens):
            global_index = old_capacity + i
            block_index = global_index // self.block_size
            offset = global_index % self.block_size
            block_id = self._request_blocks[request_id][block_index]
            slot_mapping.append(block_id * self.block_size + offset)

        current_slots.extend(slot_mapping)
        return (slot_mapping, new_block_ids)


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

        now = time.time()
        for block in self._request_blocks[request_id]:
            if block in self._block_info:
                self._block_info[block]["last_used"] = now
                self._block_info[block]["ref_count"] += 1

        # if num_new_tokens is zero, no new allocation is needed
        if num_new_tokens == 0:
            return ([], [])

        needed_blocks = (num_new_tokens + self.block_size - 1) // self.block_size

        # if insufficient free blocks exist, we do eviction
        if needed_blocks > len(self._free_gpu_blocks):
            self._evict_blocks(needed_blocks - len(self._free_gpu_blocks))

        if needed_blocks > len(self._free_gpu_blocks):
            raise RuntimeError(f"Not enough free blocks even after eviction. Needed {needed_blocks}, have {len(self._free_gpu_blocks)}")

        new_block_ids = self._free_gpu_blocks[:needed_blocks]
        self._free_gpu_blocks = self._free_gpu_blocks[needed_blocks:]
        self._request_blocks[request_id].extend(new_block_ids)

        for block in new_block_ids:
            self._block_info[block] = {"ref_count": 1, "last_used": now}

        tokens_assigned = 0
        for block_id in new_block_ids:
            tokens_in_block = min(self.block_size, num_new_tokens - tokens_assigned)
            for i in range(tokens_in_block):
                slot_mapping.append(block_id * self.block_size + i)
                tokens_assigned += 1  #TODO: optimize code to avoid appending one by one (try block-wise)
            if tokens_assigned >= num_new_tokens:
                break

        self._request_slots[request_id].extend(slot_mapping)

        # for CacheDocRequests, record in the doc mapping
        if request.get_type() == RAGRequestType.CACHE_DOC:
            cache_req = request  # type: CacheDocRequest
            self._doc_to_blocks[cache_req.doc_id] = self._request_blocks[request_id].copy()

        return (slot_mapping, new_block_ids)

    def get_computed_gpu_blocks(self, request: RAGRequest, swap_in_cpu_blocks: bool = False) -> List[int]:
        """
        Returns the list of GPU block IDs that have already been computed (cached) for the given request.
        For CacheDocRequests, if the document is cached (tracked in _doc_to_blocks), that copy is returned.
        For other requests, internal bookkeeping is used.
        If swap_in_cpu_blocks is True and swapping is enabled, attempts to swap CPU blocks into GPU memory.
        
        In either case, each returned block has its last_used timestamp updated and its ref_count incremented.
        """
        # for CacheDocRequests, check our internal doc mapping first
        if request.get_type() == RAGRequestType.CACHE_DOC:
            cache_req = request  # type: CacheDocRequest
            if cache_req.doc_id in self._doc_to_blocks:
                blocks = self._doc_to_blocks[cache_req.doc_id].copy()
                return blocks

        if request.request_id not in self._request_blocks:
            return []
        
         # TODO: the other case is for prefix check. if two requests have different req ids but the same prefix, they should share the same blocks; logic to handle  pre-empted blocks
        
        gpu_blocks = self._request_blocks[request.request_id].copy()


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
        return gpu_blocks

    def num_free_blocks(self) -> int:
        """Return the number of free GPU blocks."""
        return len(self._free_gpu_blocks)

    # TODO: evict == overwrite == remove the evicted blocks from others' _request_blocks, _request_slots, doc_to_blocks etc.
    # technically, all blocks with ref_count == 1 can be evicted
     # but it's better to evict blocks for non-cache doc requests first
     #update eviction to be block wise
    def _evict_blocks(self, num_required: int) -> None:
        """
        Evict blocks from cached documents until at least num_required free GPU blocks are available.
        The eviction policy selects documents based on the lowest average reference count and oldest usage.
        """
        while len(self._free_gpu_blocks) < num_required and self._doc_to_blocks:
            candidate_doc = None
            candidate_score = None
            for doc_id, block_list in self._doc_to_blocks.items():
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


