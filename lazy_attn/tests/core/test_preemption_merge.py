"""A document request that is preempted and resumed.

Preemption puts a request back on the waiting queue, so the scheduler's
document-handling block runs for it a second time. It used to merge the
documents into the prompt again, which grew the prompt by another full copy of
them and dropped every token generated before the preemption; the next
scheduling step then computed a negative `num_new_tokens` and the engine died on
`assert num_new_tokens > 0`.
"""
import pytest

from conftest import make_lazy_request

DOCUMENTS_TOKEN_IDS_PADDED = [[1, 2, 3, 4], [5, 6, 7, 8]]
DOCUMENT_LENS = [4, 4]
DOCUMENT_LENS_PADDED = [4, 4]
QUERY_TOKEN_IDS = [100, 101, 102]


def make_document_request():
    return make_lazy_request(
        prompt_token_ids=list(QUERY_TOKEN_IDS),
        documents_token_ids_padded=DOCUMENTS_TOKEN_IDS_PADDED,
        document_lens=DOCUMENT_LENS,
        document_lens_padded=DOCUMENT_LENS_PADDED,
    )


@pytest.mark.unit
def test_merge_documents_merges_once():
    request = make_document_request()
    assert request.merge_documents() is True

    merged = list(request.prompt_token_ids)
    assert merged == [1, 2, 3, 4, 5, 6, 7, 8] + QUERY_TOKEN_IDS
    assert request.num_prompt_tokens == len(merged)

    # The second call reports that it did nothing and leaves the prompt alone.
    assert request.merge_documents() is False
    assert list(request.prompt_token_ids) == merged
    assert request.num_prompt_tokens == len(merged)


@pytest.mark.unit
def test_merge_documents_keeps_generated_tokens():
    """The merge must not reset the token list a resumed request generated."""
    request = make_document_request()
    request.merge_documents()
    request.append_output_token_ids([200, 201])

    request.merge_documents()  # what preemption triggers

    assert list(request.output_token_ids) == [200, 201]
    assert list(request.all_token_ids) == list(request.prompt_token_ids) + [
        200, 201
    ]
    assert request.num_tokens == request.num_prompt_tokens + 2


@pytest.mark.unit
def test_resumed_request_still_has_tokens_left_to_schedule():
    """The arithmetic that used to trip `assert num_new_tokens > 0`.

    On resumption the scheduler adds the document hits to whatever
    `get_computed_blocks` found. Those hits are already covered there -- the
    document block hashes were prepended to `req_to_block_hashes` on the first
    pass -- so the documents must not be added a second time, and the prompt
    must not have grown either. Both halves are needed: this passes only if the
    merge stayed idempotent.
    """
    request = make_document_request()

    # First scheduling: merged, then it generates a token.
    request.merge_documents()
    cached_prefix = request.num_prompt_tokens
    request.append_output_token_ids([200])

    # Preemption: blocks freed, counter reset, back onto the waiting queue.
    request.num_computed_tokens = 0

    # Resumption, in the order the scheduler does it. get_computed_blocks runs
    # first and matches the whole documents+query prefix, because the document
    # block hashes were prepended to req_to_block_hashes on the first pass; it
    # caps the hit at num_tokens - 1 so a token is always left to sample from.
    num_computed_tokens = min(cached_prefix, request.num_tokens - 1)

    # Then the document block, which is where the double-count used to happen.
    # Whether it runs is keyed off the prompt actually changing rather than off
    # merge_documents()'s return value, so this reproduces the original crash
    # and not merely the contract the fix introduced.
    prompt_len_before = request.num_prompt_tokens
    request.merge_documents()
    if request.num_prompt_tokens != prompt_len_before:
        num_computed_tokens += sum(DOCUMENT_LENS_PADDED)

    num_new_tokens = request.num_tokens - num_computed_tokens

    assert num_new_tokens > 0
