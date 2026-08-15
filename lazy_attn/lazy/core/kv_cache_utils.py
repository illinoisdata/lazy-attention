"""Block hashing for requests that carry documents.

Changed by Haocheng at 2025/09/07
"""

from typing import Any

from vllm.v1.core.kv_cache_utils import (BlockHash,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens, need_extra_keys)

from lazy.request import LazyRequest as Request


def hash_request_tokens_docs(hash_function: Any, block_size: int,
                             request: Request) -> list[list[BlockHash]]:
    """The block hashes of each document, one inner list per document.

    Each document is hashed exactly as the request that populates it hashes
    itself -- by handing that request to vLLM's own `hash_request_tokens`. The
    two therefore cannot disagree: whatever upstream folds into a block hash
    (the cache salt today, whatever it adds next) is applied once, on the same
    object, for both the lookup and the write.
    """
    assert request.has_documents
    return [
        hash_request_tokens(hash_function, block_size,
                            request.document_request(doc_idx))
        for doc_idx in range(len(request.documents_token_ids_padded))
    ]


def hash_request_tokens_with_doc_hash(hash_function: Any, block_size: int,
                                      request: Request) -> list[BlockHash]:
    """`hash_request_tokens`, seeded with the document-sequence hash.

    The query blocks sit behind every document, so the chain starts from the
    documents' combined hash instead of upstream's `None`. That seed is the
    only difference -- `hash_block_tokens` takes no seed parameter, which is
    why the loop is here at all; the extra keys come from upstream.
    """
    token_ids = request.all_token_ids

    req_need_extra_keys = need_extra_keys(request)
    req_extra_keys = None
    curr_mm_idx = 0

    ret = []
    parent_block_hash_value = request.document_seq_hash
    if isinstance(parent_block_hash_value, str):
        parent_block_hash_value = int(parent_block_hash_value, 16)
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        if req_need_extra_keys:
            req_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

        block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
                                       block_token_ids, req_extra_keys)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret
