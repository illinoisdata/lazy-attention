from vllm.v1.core.kv_cache_utils import hash_request_tokens


def hash_request_tokens_no_prefix(hash_function: Any, block_size: int,
                                  request: Request) -> list[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. This version does not depend on the hash of the previous block.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids

    req_need_extra_keys = need_extra_keys(request)
    req_extra_keys = None
    curr_mm_idx = 0

    ret = []
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        if req_need_extra_keys:
            # MM and LoRA requests need extra keys for block-hash computation.
            req_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

        # Compute the hash for the current block without prefix dependency.
        block_hash = hash_block_tokens(hash_function, None,
                                       block_token_ids, req_extra_keys)
        ret.append(block_hash)
    return ret