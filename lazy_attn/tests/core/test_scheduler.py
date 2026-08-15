# Test the scheduling logic

from itertools import chain

import pytest

from conftest import make_lazy_request
from lazy.core.sched.scheduler import (metadata_for_lazy_attention,
                                       metadata_for_mepic)

BLOCK_SIZE = 8

# Four documents, right-padded to a whole number of blocks. The padding is what
# the rotation metadata has to account for: real lengths [6, 15, 8, 2] padded
# to [8, 16, 8, 8], i.e. 5 document blocks in total.
DOCUMENTS_TOKEN_IDS_PADDED = [
    [1, 2, 3, 4, 5, 6, 128001, 128001],
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 128001],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 128001, 128001, 128001, 128001, 128001, 128001],
]
DOCUMENT_LENS = [6, 15, 8, 2]
DOCUMENT_LENS_PADDED = [8, 16, 8, 8]


@pytest.fixture
def mock_request():
    return make_lazy_request(
        eos_token_id=128001,
        documents_token_ids_padded=DOCUMENTS_TOKEN_IDS_PADDED,
        document_lens=DOCUMENT_LENS,
        document_lens_padded=DOCUMENT_LENS_PADDED,
    )


@pytest.mark.unit
def test_metadata_for_lazy_attention(mock_request):
    q_offset, q_mask = metadata_for_lazy_attention(mock_request, BLOCK_SIZE)

    # One entry per document block, plus one for the query/decode block.
    #
    # q_offset: every block of a document repeats that document's rotation
    # offset, biased by +1 so 0 stays free as a sentinel. Document i rotates by
    # the total padding (9) plus the real length of every earlier document:
    #   doc0 -> 9,  doc1 -> 9+6=15,  doc2 -> 15+15=30,  doc3 -> 30+8=38
    # and the trailing 1 resets the query block to the global RoPE orientation.
    #
    # q_mask: the padding of each document, recorded on its last block only.
    assert q_offset == [10, 16, 16, 31, 39, 1]
    assert q_mask == [2, 0, 1, 0, 6, 0]


@pytest.mark.unit
def test_metadata_for_mepic_zeroes_offsets(mock_request):
    # MEPIC rotates cached keys inside the attention kernel, so the scheduler
    # hands it no rotation offsets -- only the padding mask.
    _, lazy_mask = metadata_for_lazy_attention(mock_request, BLOCK_SIZE)
    q_offset, q_mask = metadata_for_mepic(mock_request, BLOCK_SIZE)

    assert q_offset == [0] * len(q_offset)
    assert q_mask == lazy_mask


@pytest.mark.unit
def test_documents_merge_in_front_of_the_prompt(mock_request):
    query_tokens = list(mock_request.prompt_token_ids)
    mock_request.merge_documents()

    assert mock_request.prompt_token_ids == (
        list(chain.from_iterable(DOCUMENTS_TOKEN_IDS_PADDED)) + query_tokens)
    assert mock_request.num_prompt_tokens == len(mock_request.prompt_token_ids)
    assert list(mock_request.all_token_ids) == mock_request.prompt_token_ids


@pytest.mark.unit
def test_document_request_carries_the_hashing_fields(mock_request):
    mock_request.cache_salt = "tenant-a"
    mock_request.sampling_params.max_tokens = 64
    doc = mock_request.document_request(1)

    assert doc.request_id == f"{mock_request.request_id}_d1"
    assert doc.prompt_token_ids == DOCUMENTS_TOKEN_IDS_PADDED[1]
    assert doc.is_document_request
    # Everything that feeds a block hash has to match the parent, or the
    # parent can never find the blocks this request writes.
    assert doc.cache_salt == "tenant-a"
    # A document is prefill-only; it never samples. The parent's own params
    # must survive that -- they are shared until deep-copied.
    assert doc.sampling_params.max_tokens == 1
    assert mock_request.sampling_params.max_tokens == 64
