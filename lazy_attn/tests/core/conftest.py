"""Shared construction helpers for the core (CPU-only) tests."""
import pytest

from lazy.request import LazyRequest


def make_lazy_request(**overrides) -> LazyRequest:
    """A LazyRequest with everything irrelevant filled in.

    `LazyRequest.__init__` mirrors vLLM's `Request.__init__` and shifts on
    every port, so the boilerplate lives in one place -- each test then names
    only the fields it actually cares about.
    """
    from vllm import SamplingParams

    kwargs = dict(
        request_id="test_request",
        prompt_token_ids=[100, 101, 102, 103],
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=SamplingParams(max_tokens=1),
        eos_token_id=None,
        arrival_time=0.0,
    )
    kwargs.update(overrides)
    return LazyRequest(**kwargs)


@pytest.fixture
def lazy_request_factory():
    return make_lazy_request
