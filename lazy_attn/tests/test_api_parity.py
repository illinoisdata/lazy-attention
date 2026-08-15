"""Guard the seams where LazyAttention re-declares vLLM's own data structures.

`LazyRequest`, the lazy `EngineCoreRequest` and the lazy `CachedRequestState`
are standalone re-declarations rather than subclasses, so a field or property
added upstream silently goes missing until something explodes deep in the
engine. These tests fail loudly on the next vLLM bump instead.

They also pin the patch registry: every symbol LazyAttention swaps out must
still exist in the installed vLLM.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

import lazy.__vllm__  # noqa: F401  applies the patches
from lazy.vllm_patch import (attention_targets, frontend_targets,
                             scheduler_targets)
from lazy.core.sched.scheduler import LazyScheduler

# The patch registry records the originals it replaced, so compare against
# those rather than re-importing (which would hand back the lazy versions).
from lazy.vllm_patch import _ORIGINALS


def _original(key: str):
    obj = _ORIGINALS.get(key)
    if obj is None:
        pytest.skip(f"no recorded original for {key}")
    return obj


@pytest.mark.unit
def test_all_patch_targets_resolve():
    """Every patched symbol must exist in the installed vLLM."""
    targets = (attention_targets() + frontend_targets()
               + scheduler_targets(LazyScheduler))
    missing = []
    for target in targets:
        try:
            owner = importlib.import_module(target.module_name)
        except ImportError as exc:
            missing.append(f"{target.key} (module: {exc})")
            continue
        for part in target.attribute_name.split("."):
            if not hasattr(owner, part):
                missing.append(f"{target.key} (missing {part!r})")
                break
            owner = getattr(owner, part)
    assert not missing, "patch targets no longer present in vLLM:\n  " + \
        "\n  ".join(missing)


# Methods the lazy classes override, and therefore have to keep accepting
# everything upstream's callers pass. vLLM adds request-routing parameters to
# these between releases -- `tokenization_kwargs` and `data_parallel_rank` both
# landed in 0.9.x -- and a missing one is a TypeError raised from inside vLLM's
# own frontend, on a path the lazy tests do not otherwise reach.
OVERRIDDEN_METHODS = [
    ("vllm.entrypoints.llm.LLM", "lazy.entrypoints.llm", "LazyLLM",
     ["generate", "_validate_and_add_requests", "_add_request"]),
    ("vllm.v1.engine.llm_engine.LLMEngine", "lazy.engine.llm_engine",
     "LazyLLMEngine", ["add_request"]),
    ("vllm.v1.engine.async_llm.AsyncLLM", "lazy.engine.async_llm",
     "AsyncLazyLLM", ["add_request", "generate"]),
]


@pytest.mark.unit
@pytest.mark.parametrize("origin_key,module_name,cls_name,method_names",
                         OVERRIDDEN_METHODS,
                         ids=[m[2] for m in OVERRIDDEN_METHODS])
def test_overrides_accept_every_upstream_parameter(origin_key, module_name,
                                                   cls_name, method_names):
    """Every parameter upstream accepts must survive the override."""
    original_cls = _original(origin_key)
    lazy_cls = getattr(importlib.import_module(module_name), cls_name)

    missing = []
    for method_name in method_names:
        original = getattr(original_cls, method_name, None)
        if original is None:
            continue  # upstream dropped it; nothing to keep parity with
        expected = set(inspect.signature(original).parameters)
        actual = set(
            inspect.signature(getattr(lazy_cls, method_name)).parameters)
        for param in sorted(expected - actual):
            missing.append(f"{cls_name}.{method_name}(): {param}")

    assert not missing, (
        "lazy overrides dropped parameters vLLM's own callers pass:\n  " +
        "\n  ".join(missing))


@pytest.mark.unit
def test_lazy_request_covers_vllm_request():
    """LazyRequest must expose everything vLLM's Request does."""
    vllm_request = _original("vllm.v1.request.Request")
    from lazy.request import LazyRequest

    expected = {name for name in dir(vllm_request) if not name.startswith("_")}
    actual = {name for name in dir(LazyRequest) if not name.startswith("_")}
    missing = sorted(expected - actual)
    assert not missing, (
        "LazyRequest is missing attributes vLLM's Request has: "
        f"{missing}. Add them in lazy/request.py.")


@pytest.mark.unit
def test_pooling_request_has_no_structured_output():
    """Pooling requests carry no sampling_params, and the scheduler still asks
    every request whether it uses structured output."""
    from vllm.pooling_params import PoolingParams

    from lazy.request import LazyRequest

    request = LazyRequest(
        request_id="pool",
        prompt_token_ids=[1, 2, 3],
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=None,
        pooling_params=PoolingParams(),
        eos_token_id=None,
        arrival_time=0.0,
    )
    assert request.use_structured_output is False


@pytest.mark.unit
def test_engine_core_request_covers_vllm_fields():
    """The lazy EngineCoreRequest must carry every upstream field."""
    vllm_ecr = _original("vllm.v1.engine.EngineCoreRequest")
    from lazy.engine import EngineCoreRequest as LazyECR

    expected = set(vllm_ecr.__struct_fields__)
    actual = set(LazyECR.__struct_fields__)
    missing = sorted(expected - actual)
    assert not missing, (
        f"lazy EngineCoreRequest is missing fields: {missing}. "
        "Add them in lazy/engine/__init__.py.")


@pytest.mark.unit
def test_cached_request_state_covers_vllm_fields():
    """The lazy CachedRequestState must carry every upstream field."""
    vllm_crs = _original("vllm.v1.worker.gpu_input_batch.CachedRequestState")
    from lazy.worker.gpu_input_batch import CachedRequestState as LazyCRS

    expected = {f.name for f in dataclasses.fields(vllm_crs)}
    actual = {f.name for f in dataclasses.fields(LazyCRS)}
    missing = sorted(expected - actual)
    assert not missing, (
        f"lazy CachedRequestState is missing fields: {missing}. "
        "Add them in lazy/worker/gpu_input_batch.py.")
