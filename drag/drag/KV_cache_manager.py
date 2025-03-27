from typing import List, Tuple

from torch import Tensor

from torch import Tensor
from drag import RAGRequest


class KV_Cache_Manager:
    BLOCK_SIZE = 16 #number of tokens in a block
    BLOCK_SIZE = 16 #number of tokens in a block
    enable_swap_in_cpu_blocks = False
    def __init__(self, block_size: int, gpu_kv_cache: List[Tensor], cpu_kv_cache: List[Tensor], num_gpu_blocks: int, num_cpu_blocks: int):
        pass
    def free(request: RAGRequest) -> None:
        pass

    def append_slots(request: RAGRequest, num_new_tokens: int) -> Tuple[List[int], List[int]]:
        slot_mapping = []
        new_block_ids = []
        pass

    def __init__(self, total_blocks=0):

        # track GPU blocks (free and allocated)
        self._free_gpu_blocks = list(range(total_blocks))

        # track blocks stored in CPU memory
        self._cpu_blocks = {}

        # track allocated blocks for each request
        self._request_blocks = {}

        # track slot mappings for each request
        self._request_slots = {}

        # total number of GPU blocks available
        self._total_gpu_blocks = total_blocks
    

    def free(self,request: RAGRequest) -> None:
        request_id = request.request_id
        
        if request_id not in self._request_blocks:
            return  # request not registered or already freed
        
        # return allocated blocks to the free pool
        self._free_gpu_blocks.extend(self._request_blocks[request_id])
        
        # clean up request data
        del self._request_blocks[request_id]
        if request_id in self._request_slots:
            del self._request_slots[request_id]
        if request_id in self._cpu_blocks:
            del self._cpu_blocks[request_id]
        pass

    def append_slots(self, request: RAGRequest, num_new_tokens: int) -> List[int]:
        #return the slot mapping for the new tokens
        # throw error if not enough free blocks to allocate the new tokens. This means scheduler has logical error because it should've checked the free blocks before sending the request.
        return slot_mapping, new_block_ids

    def get_computed_gpu_blocks(request: RAGRequest, swap_in_cpu_blocks=False) -> List[int]:
        pass
        # return the id of blocks that are already computed in gpu. 
        # if swap_in_cpu_blocks is True, also swap in those blocks storing in dram to the gpu memory
        # then the returned blocks should also include those just swapped in from dram

        # no need to increase the block_counts for these blocks.
        # throw error if can't swap in cpu blocks. This means scheduler has logical error because it should've checked the free blocks before sending the request.

        request_id = request.request_id
        
        # empty list if request not found
        if request_id not in self._request_blocks:
            return []
        
        # get blocks already in GPU
        gpu_blocks = self._request_blocks[request_id].copy()
        
        # handleing CPU blocks if swapping enabled
        if swap_in_cpu_blocks and self.enable_swap_in_cpu_blocks and request_id in self._cpu_blocks:
            cpu_blocks = self._cpu_blocks[request_id]
            
            if len(cpu_blocks) > len(self._free_gpu_blocks):
                raise RuntimeError(f"Not enough free blocks to swap in. Need {len(cpu_blocks)}, have {len(self._free_gpu_blocks)}")
            
            # allocate GPU blocks for the CPU data
            new_gpu_blocks = self._free_gpu_blocks[:len(cpu_blocks)]
            self._free_gpu_blocks = self._free_gpu_blocks[len(cpu_blocks):]
            
            # update request's blocks
            self._request_blocks[request_id].extend(new_gpu_blocks)
            gpu_blocks.extend(new_gpu_blocks)
            
            #empty cpu blocks
            del self._cpu_blocks[request_id]
        
        return gpu_blocks

    def num_free_blocks(self) -> int:
        # num of free gpu blocks
        return len(self._free_gpu_blocks)

    def allocate_slots(self, request: RAGRequest, num_new_tokens:int) -> Tuple[List[int], List[int]]:
        #return the slot mapping for the new tokens
        # if num_new_tokens = 0. return 2 empty lists. But the request should be registered as running request, and the block_counts should be increased.
        # throw error if not enough free blocks to allocate the new tokens. This means scheduler has logical error because it should've checked the free blocks before sending the request.
        request_id = request.request_id
        slot_mapping = []
        new_block_ids = []
        
        # add request
        if request_id not in self._request_blocks:
            self._request_blocks[request_id] = []
        if request_id not in self._request_slots:
            self._request_slots[request_id] = []
        
        # for zero tokens, just return empty lists
        if num_new_tokens <= 0:
            return slot_mapping, new_block_ids
        
        #needed blocks
        needed_blocks = (num_new_tokens + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        
        # check if we have enough free blocks
        if needed_blocks > len(self._free_gpu_blocks):
            raise RuntimeError(f"Not enough free blocks. Need {needed_blocks}, have {len(self._free_gpu_blocks)}")
        
        # allocate blocks
        new_block_ids = self._free_gpu_blocks[:needed_blocks]
        self._free_gpu_blocks = self._free_gpu_blocks[needed_blocks:]
        self._request_blocks[request_id].extend(new_block_ids)
        
        # slot mapping for the tokens
        tokens_assigned = 0
        
        for block_id in new_block_ids:
            # tokens for this block
            tokens_in_block = min(self.BLOCK_SIZE, num_new_tokens - tokens_assigned)
            
            # slots in this block
            for i in range(tokens_in_block):
                slot_mapping.append(block_id * self.BLOCK_SIZE + i)
                tokens_assigned += 1
                
            if tokens_assigned >= num_new_tokens:
                break
        
        # slot mappings
        self._request_slots[request_id].extend(slot_mapping)
        
        return slot_mapping, new_block_ids

