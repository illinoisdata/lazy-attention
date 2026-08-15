"""
We change this to add extra fields in EngineCoreRequest for LazyAttention.

Changed by Haocheng at 2025/09/04
"""

import enum
import time
from collections.abc import Sequence
from typing import Any, Optional, Union

import msgspec

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""
    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3
    # TODO(haocheng): add more detailed events for lazy attention
    
    
class EngineCoreRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    # NOTE: prompt and prompt_token_ids should be DecoderOnlyInput,
    # but this object is currently not playing well with msgspec
    # due to circular imports and typing we have in data.py

    request_id: str
    prompt_token_ids: list[int]
    mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]]
    mm_hashes: Optional[list[str]]
    mm_placeholders: Optional[list[PlaceholderRange]]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
    cache_salt: Optional[str]
    data_parallel_rank: Optional[int]

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    # Extra arguments for lazy attention.
    # These are defaulted so that vLLM's own call sites -- which know nothing
    # about documents -- can still construct an EngineCoreRequest once this
    # class is patched in.
    documents_token_ids_padded: Optional[list[list[int]]] = None
    document_seq_hash: Optional[str] = None
    document_lens: Optional[list[int]] = None
    document_lens_padded: Optional[list[int]] = None


def apply_patch():
    import vllm.v1.engine.__init__
    vllm.v1.engine.__init__.EngineCoreRequest = EngineCoreRequest
