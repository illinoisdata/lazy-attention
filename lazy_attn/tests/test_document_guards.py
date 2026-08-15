"""Request shapes the document path cannot serve, rejected at the door.

Both are checked in `LazyProcessor`, the one point every entry passes through,
because both fail badly if they get past it: a pooling parent raises
`AttributeError` from inside the scheduler when its document request is
spawned, and a LoRA parent does not fail at all -- it quietly attends to
document KV computed by the base model.
"""
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
