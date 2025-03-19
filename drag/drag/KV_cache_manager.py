from typing import List, Tuple
from drag import RAGRequest


class KV_Cache_Manager:
    BLOCK_SIZE = 128 #number of tokens in a block
    enable_swap_in_cpu_blocks = False
    def free(request: RAGRequest) -> None:
        pass

    def append_slots(request: RAGRequest, num_new_tokens: int) -> List[int]:
        pass
        #return the slot mapping for the new tokens
        # return non if not enough free blocks to allocate the new tokens.

    def get_computed_gpu_blocks(request: RAGRequest, swap_in_cpu_blocks=False) -> List[int]:
        pass
        # return the id of blocks that are already computed in gpu. 
        # if swap_in_cpu_blocks is True, also swap in those blocks storing in dram to the gpu memory
        # then the returned blocks should also include those just swapped in from dram

        # no need to increase the block_counts for these blocks.
        # throw error if can't swap in cpu blocks. This means scheduler has logical error because it should've checked the free blocks before sending the request.

    def num_free_blocks() -> int:
        # num of free gpu blocks
        pass

    def allocate_slots(request: RAGRequest, num_new_tokens:int) -> Tuple[List[int], List[int]]:
        slot_mapping = []
        new_block_ids = []
        pass
        #return the slot mapping for the new tokens
        # if num_new_tokens = 0. return 2 empty lists. But the request should be registered as running request, and the block_counts should be increased.
        # throw error if not enough free blocks to allocate the new tokens. This means scheduler has logical error because it should've checked the free blocks before sending the request.
        return slot_mapping, new_block_ids


    