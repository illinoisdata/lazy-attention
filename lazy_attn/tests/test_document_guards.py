"""Request shapes the document path cannot serve, rejected at the door.

All three are checked in `LazyProcessor`, the one point every entry passes
through, because all three fail badly if they get past it: a pooling parent
raises `AttributeError` from inside the scheduler when its document request is
spawned, a LoRA parent does not fail at all -- it quietly attends to document
KV computed by the base model -- and a prompt-logprobs parent has prefix
caching disabled under it, so once it is rescheduled it prefills the merged
prompt and writes cross-document KV under the per-document cache hashes.
"""
from types import SimpleNamespace

import pytest

from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

from lazy.engine.processor import _validate_document_request


@pytest.mark.unit
def test_sampling_request_without_lora_is_accepted():
    _validate_document_request(SamplingParams(max_tokens=4), None)


@pytest.mark.unit
def test_pooling_request_with_documents_is_rejected():
    with pytest.raises(ValueError, match="pooling"):
        _validate_document_request(PoolingParams(), None)


@pytest.mark.unit
def test_lora_request_with_documents_is_rejected():
    lora = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path="/tmp/x")
    with pytest.raises(ValueError, match="LoRA"):
        _validate_document_request(SamplingParams(max_tokens=4), lora)


@pytest.mark.unit
def test_prompt_logprobs_request_with_documents_is_rejected():
    with pytest.raises(ValueError, match="prompt_logprobs"):
        _validate_document_request(
            SamplingParams(max_tokens=4, prompt_logprobs=1), None)


@pytest.mark.unit
def test_prompt_logprobs_turns_the_prefix_hit_off():
    """Why the guard exists: the hit a rescheduled request would rely on.

    `get_computed_blocks` returns nothing for a prompt-logprobs request, by
    design -- upstream wants every prompt token recomputed for its logits. The
    documents are merged into the prompt exactly once, so on the second pass
    (resumed after preemption, or retried after a failed allocation) the
    document branch is skipped too and the whole merged prompt is prefilled as
    one stream, under the per-document hashes. Pinned here so the guard is not
    removed on the assumption that some other path would catch it.
    """
    from lazy.core.kv_cache_manager import LazyKVCacheManager

    manager = SimpleNamespace(
        enable_caching=True,
        create_empty_block_list=lambda: "empty",
    )
    request = SimpleNamespace(
        sampling_params=SamplingParams(max_tokens=4, prompt_logprobs=1))
    blocks, num_computed = LazyKVCacheManager.get_computed_blocks(
        manager, request)
    assert (blocks, num_computed) == ("empty", 0)


@pytest.mark.unit
def test_pooling_parent_would_have_crashed_spawning_a_document():
    """Why the pooling guard exists: `document_request` needs sampling_params.

    Kept as an executable statement of the failure being prevented -- if the
    spawn ever stops needing them, this test says so.
    """
    from lazy.request import LazyRequest

    request = LazyRequest(
        request_id="pool",
        prompt_token_ids=[100, 101],
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=None,
        pooling_params=PoolingParams(),
        eos_token_id=None,
        arrival_time=0.0,
        documents_token_ids_padded=[[1, 2, 3, 4]],
        document_lens=[4],
        document_lens_padded=[4],
    )
    with pytest.raises(AttributeError):
        request.document_request(0)
