from typing import Any

from vllm.v1.core.kv_cache_utils import need_extra_keys, generate_block_hash_extra_keys, hash_block_tokens
from vllm.v1.core.kv_cache_utils import BlockHashType

from lazy.request import LazyRequest as Request



def hash_request_tokens_docs(hash_function: Any, block_size: int,
                                  request: Request) -> list[list[BlockHashType]]:
    """Compute the hash values for each document in the document sequence.
    Note the the return value is a list of lists, where each inner list contains
    the hash values for a single document."""
    documents_token_ids = request.documents_token_ids

    ret = []
    for doc_idx, token_ids in enumerate(documents_token_ids):
        ret.append([])
        parent_block_hash_value = None
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
                                        block_token_ids, None)
            ret[doc_idx].append(block_hash)
            parent_block_hash_value = block_hash.hash_value
    return ret


def hash_request_tokens_with_doc_hash(hash_function: Any, block_size: int,
                                      request: Request) -> list[BlockHashType]:
    """The only difference between this function and the original one is that
    the hash of the document sequence is used as the prefix for the block hash."""
    token_ids = request.all_token_ids

    ret = []
    parent_block_hash_value = request.document_seq_hash
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
                                       block_token_ids, None)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret

# original_hash_request_tokens = None

# def apply_patch():
#     global original_hash_request_tokens
#     import vllm.v1.core.kv_cache_utils
#     original_hash_request_tokens = vllm.v1.core.kv_cache_utils.hash_request_tokens
#     vllm.v1.core.kv_cache_utils.hash_request_tokens = hash_request_tokens_no_prefix

# def revert_patch():
#     import vllm.v1.core.kv_cache_utils
#     vllm.v1.core.kv_cache_utils.hash_request_tokens = original_hash_request_tokens