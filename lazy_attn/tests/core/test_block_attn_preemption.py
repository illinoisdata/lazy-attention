"""Block-Attention refuses a preempted document request, before scheduling it.

The refusal itself is the placeholder for unimplemented work: copy-on-write
only rotates blocks for freshly-scheduled requests, so a resumed one would
attend to canonical, unrotated document keys with a query the runner has
already pinned to `q_offset = 1`. Wrong, and silent.

What these tests pin is *when* it refuses. The check reads the waiting queue
before `schedule()` delegates upstream, so it fires a step before the request
would be resumed and while the scheduler still holds no state for it. Refusing
after the fact -- once blocks are allocated, the request is in `running` and its
computed-token count has advanced for a forward pass that will now never
happen -- leaves the scheduler describing work that did not occur.
"""
from types import SimpleNamespace

import pytest

from vllm.v1.request import RequestStatus

from block_attn_vllm.scheduler import BlockAttnScheduler
from lazy.core.sched.scheduler import LazyScheduler


def _request(request_id: str, status, has_documents: bool):
    return SimpleNamespace(request_id=request_id, status=status,
                           has_documents=has_documents)


def _scheduler(*requests) -> BlockAttnScheduler:
    """A scheduler with a waiting queue and nothing else -- no engine needed."""
    scheduler = object.__new__(BlockAttnScheduler)
    scheduler.waiting = list(requests)
    return scheduler


@pytest.mark.unit
def test_preempted_document_request_is_rejected():
    scheduler = _scheduler(
        _request("q0", RequestStatus.PREEMPTED, has_documents=True))
    with pytest.raises(NotImplementedError, match="q0"):
        scheduler._reject_preempted_lazy_requests()


@pytest.mark.unit
def test_ordinary_waiting_requests_are_left_alone():
    scheduler = _scheduler(
        _request("fresh", RequestStatus.WAITING, has_documents=True),
        _request("no-docs", RequestStatus.PREEMPTED, has_documents=False),
    )
    scheduler._reject_preempted_lazy_requests()


@pytest.mark.unit
def test_nothing_is_scheduled_before_the_refusal(monkeypatch):
    """The point of checking the queue rather than the SchedulerOutput.

    If this ever regresses to inspecting the output, `schedule` runs first and
    the exception arrives after blocks are allocated and the request has been
    moved to `running` -- unwindable only by the caller, which does not.
    """
    delegated = False

    def record(self):
        nonlocal delegated
        delegated = True
        raise AssertionError("upstream schedule() should not have been reached")

    monkeypatch.setattr(LazyScheduler, "schedule", record)
    scheduler = _scheduler(
        _request("q0", RequestStatus.PREEMPTED, has_documents=True))

    with pytest.raises(NotImplementedError):
        scheduler.schedule()
    assert not delegated
