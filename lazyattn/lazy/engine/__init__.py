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
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
    
    # Extra arguments for lazy attention
    documents_token_ids_padded: Optional[list[list[int]]]
    document_seq_hash: Optional[str]
    document_lens: Optional[list[int]]
    document_lens_padded: Optional[list[int]]

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0


def apply_patch():
    import vllm.v1.engine.__init__
    vllm.v1.engine.__init__.EngineCoreRequest = EngineCoreRequest
