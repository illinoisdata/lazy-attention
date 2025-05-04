# SPDX-License-Identifier: Apache-2.0

import enum
import time
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
    DOC_QUEUED = 4
    QUERY_QUEUED = 5


class EngineCoreRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    # NOTE: prompt and prompt_token_ids should be DecoderOnlyInput,
    # but this object is currently not playing well with msgspec
    # due to circular imports and typing we have in data.py

    request_id: str
    # NOTE(ywang96): original text prompt is needed when a request is added to
    # Detokenizer, but set to None when it is added to EngineCoreClient.
    prompt: Optional[str]
    prompt_token_ids: list[int]
    mm_inputs: Optional[list[MultiModalKwargs]]
    mm_hashes: Optional[list[str]]
    mm_placeholders: Optional[list[PlaceholderRange]]
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
    # Extra arguments for dynamic rag
    documents_token_ids: Optional[list[list[int]]]
    documents_hash: Optional[list[str]]
    document_seq_hash: Optional[str]
    

def apply_patch():
    import vllm.v1.engine.__init__
    vllm.v1.engine.__init__.EngineCoreRequest = EngineCoreRequest