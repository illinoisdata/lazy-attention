# Test the schedling logic


from lazy.core.sched.scheduler import metadata_for_lazy_attention

from itertools import chain
import pytest
import numpy as np

@pytest.fixture
def mock_request():
    documents_token_ids = [[1, 2, 3, 4, 5, 6, 128001, 128001], 
                             [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 128001],
                             [1, 2, 3, 4, 5, 6, 7, 8],
                             [1, 2, 128001, 128001, 128001, 128001, 128001, 128001]]
    num_padding_tokens = [2, 1, 0, 6]
    len_documents = [8, 16, 8, 8]
    all_token_ids = list(chain.from_iterable(documents_token_ids))

    return Request(
        request_id="test_request",
        documents_token_ids=documents_token_ids,
        num_padding_tokens=num_padding_tokens,
        len_documents=len_documents,
        all_token_ids=all_token_ids,
    )

def test_metadata_for_lazy_attention(mock_request):
    block_size = 8
    q_offset, q_mask = metadata_for_lazy_attention(mock_request, block_size)
    assert q_offset == [0, 0, 8, 8, 16, 16, 24, 24, 32]
    assert q_mask == [2, 0, 1, 0, 6]