# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
    from vllm.sampling_params import SamplingParams
    # from vllm.v1.request import Request
    from minidrag.request import _Request as Request

@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[MultiModalKwargs]
    mm_hashes: list[str]
    mm_positions: list[PlaceholderRange]
    sampling_params: SamplingParams
    block_ids: list[int]
    num_computed_tokens: int
    lora_request: Optional[LoRARequest]
    
    # /////////
    # For q rotation
    has_docs: bool = False
    q_offset: Optional[list[int]] = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: list[int],
        # ///////////
        q_offset: Optional[list[int]] = None,
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            has_docs=request.has_documents,
            q_offset=q_offset,
        )