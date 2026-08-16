"""What survives a request being scheduled out of the waiting queue twice.

Two things are keyed off "is this the request's first pass", and only one of
them may be:

* the document merge and its accounting -- once, or the documents are counted
  twice;
* the rotation metadata -- *every* pass, because a request can be merged and
  then fail to be scheduled, and the metadata dictionaries do not outlive the
  `schedule()` call that built them.

The decision is exercised directly: standing up a real `Scheduler` needs vLLM's
own test helpers, which ship only in a source checkout.
"""
import pytest

from conftest import make_lazy_request
from lazy.core.sched.scheduler import metadata_for_variant

BLOCK_SIZE = 4
DOCUMENTS = [[1, 2, 3, 4], [5, 6, 7, 8]]


def make_request():
    return make_lazy_request(
        prompt_token_ids=[100, 101],
        documents_token_ids_padded=DOCUMENTS,
        document_lens=[4, 4],
        document_lens_padded=[4, 4],
    )


def schedule_pass(request, metadata):
    """The scheduler's decision for one waiting-queue pass.

    `metadata` stands in for the per-`schedule()` `req_to_q_offset` dictionary.
    Returns whether the document block ran -- the block that merges, counts the
    document hits and applies `drop_first_cached_block`.
    """
    just_merged = request.has_documents and request.merge_documents()
    if request.has_documents:
        metadata[request.request_id] = metadata_for_variant(
            request, BLOCK_SIZE)
    return just_merged


@pytest.mark.unit
def test_the_document_block_runs_once():
    request = make_request()
    assert schedule_pass(request, {}) is True
    assert schedule_pass(request, {}) is False


@pytest.mark.unit
def test_metadata_survives_a_failed_scheduling_attempt():
    """Merged, then out of blocks: the retry must still carry its rotation.

    `allocate_slots` returning None leaves the request in the waiting queue
    with `documents_merged` already set. Keying the metadata off the merge sent
    the request to the worker with `q_offset=None` -- the buffer row stays
    zeroed, every block reads sentinel 0 ("keep the current rotation"), and Q is
    never de-rotated. Silently, and only under memory pressure.
    """
    request = make_request()

    first_attempt = {}
    schedule_pass(request, first_attempt)
    # ... allocate_slots() returns None; the request stays in waiting and this
    # dictionary is discarded with the schedule() call that built it.

    retry = {}
    ran_document_block = schedule_pass(request, retry)

    assert not ran_document_block  # the merge is not repeated
    assert retry[request.request_id] == first_attempt[request.request_id]
    assert retry[request.request_id][0], "rotation offsets must not be empty"


@pytest.mark.unit
def test_metadata_is_the_same_on_every_pass():
    """It is a pure function of the document lengths, so it cannot drift."""
    request = make_request()
    passes = [{} for _ in range(3)]
    for metadata in passes:
        schedule_pass(request, metadata)

    assert (passes[0][request.request_id] == passes[1][request.request_id] ==
            passes[2][request.request_id])
