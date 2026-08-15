"""Hashing rules the per-document KV cache depends on.

The lazy paths hash documents outside vLLM's `hash_request_tokens`, so the
properties upstream gets for free -- cache-salt isolation, and agreement
between the hash a parent computes for a document and the hash the spawned
document request writes -- are checked here.
"""
import pytest

from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import hash_request_tokens

from conftest import make_lazy_request
from lazy.core.kv_cache_utils import (hash_request_tokens_docs,
                                      hash_request_tokens_with_doc_hash)
from lazy.engine.processor import _hash_document_seq

BLOCK_SIZE = 4
DOCUMENTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]]


def make_request(**overrides):
    kwargs = dict(
        documents_token_ids_padded=DOCUMENTS,
        document_lens=[len(d) for d in DOCUMENTS],
        document_lens_padded=[len(d) for d in DOCUMENTS],
        document_seq_hash="abc123",
    )
    kwargs.update(overrides)
    return make_lazy_request(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_salt_isolates_document_hashes(hash_fn):
    """Two requests with different salts must not share document blocks."""
    unsalted = hash_request_tokens_docs(hash_fn, BLOCK_SIZE, make_request())
    salted = hash_request_tokens_docs(hash_fn, BLOCK_SIZE,
                                      make_request(cache_salt="tenant-a"))
    other = hash_request_tokens_docs(hash_fn, BLOCK_SIZE,
                                     make_request(cache_salt="tenant-b"))

    for doc_idx in range(len(DOCUMENTS)):
        assert salted[doc_idx] != unsalted[doc_idx]
        assert salted[doc_idx] != other[doc_idx]

    # Same salt, same documents -> same hashes, or nothing is ever reused.
    again = hash_request_tokens_docs(
        hash_fn, BLOCK_SIZE,
        make_request(cache_salt="tenant-a", request_id="other_request"))
    assert again == salted


@pytest.mark.unit
@pytest.mark.parametrize("cache_salt", [None, "tenant-a"])
@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_document_hashes_match_the_spawned_request(hash_fn, cache_salt):
    """The parent's per-document hashes must equal what the document request
    writes to the cache -- otherwise the parent never sees a hit and keeps
    respawning the document."""
    parent = make_request(cache_salt=cache_salt)
    by_parent = hash_request_tokens_docs(hash_fn, BLOCK_SIZE, parent)

    for doc_idx in range(len(DOCUMENTS)):
        # The real spawn path, as the scheduler calls it.
        doc_request = parent.document_request(doc_idx)
        assert by_parent[doc_idx] == hash_request_tokens(
            hash_fn, BLOCK_SIZE, doc_request)


@pytest.mark.unit
@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_salt_isolates_query_hashes(hash_fn):
    """The query blocks chained behind the document-sequence hash carry the
    salt too."""
    unsalted = hash_request_tokens_with_doc_hash(hash_fn, BLOCK_SIZE,
                                                 make_request())
    salted = hash_request_tokens_with_doc_hash(
        hash_fn, BLOCK_SIZE, make_request(cache_salt="tenant-a"))

    assert unsalted and salted
    assert salted != unsalted


@pytest.mark.unit
def test_document_seq_hash_encodes_the_split():
    """The seed must distinguish document sets that flatten to the same tokens.

    One 8-token document and two 4-token ones hold the same tokens but are
    encoded into different block-diagonal KV, so sharing a seed would let a
    query block computed against one be reused for the other.
    """
    one_document = _hash_document_seq([[1, 2, 3, 4, 5, 6, 7, 8]])
    two_documents = _hash_document_seq([[1, 2, 3, 4], [5, 6, 7, 8]])
    three_documents = _hash_document_seq([[1, 2], [3, 4], [5, 6, 7, 8]])

    assert len({one_document, two_documents, three_documents}) == 3


@pytest.mark.unit
def test_document_seq_hash_is_stable():
    """...and identical document sets must agree, or nothing is ever reused."""
    documents = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert _hash_document_seq(documents) == _hash_document_seq(
        [list(doc) for doc in documents])
    assert _hash_document_seq(None) is None
    assert _hash_document_seq([]) is None


@pytest.mark.unit
@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_query_hashes_are_seeded_by_the_document_sequence(hash_fn):
    """Two requests with the same query but different documents must not share
    query blocks -- that seed is the whole point of this hash path."""
    one = hash_request_tokens_with_doc_hash(hash_fn, BLOCK_SIZE,
                                            make_request())
    two = hash_request_tokens_with_doc_hash(
        hash_fn, BLOCK_SIZE, make_request(document_seq_hash="def456"))

    assert one and two
    assert one != two
