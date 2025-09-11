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

    from lazy.request import LazyRequest as Request # Replace the original V1 Request

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
    
    # Lazy attention, if one req has documents, it is lazy
    is_lazy: bool = False  # [num_seqs]
    q_offset: Optional[list[int]] = None # [num_seqs, num_blocks]
    q_mask: Optional[list[int]] = None # [num_seqs, num_blocks]

    # Block attention
    need_to_copy: bool = False # Whether need to copy the blocks from old position to new position
    mapping_old_to_new_block_ids: dict[int, int] = None # If need_to_copy is True, this is a mapping from old block id to new block id
    mapping_real_to_desired_position_offset: dict[int, int] = None # If need_to_copy is True, this is a mapping from real position offset to desired position offset

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: list[int],
        q_offset: Optional[list[int]] = None,
        q_mask: Optional[list[int]] = None,
        need_to_copy: bool = False,
        mapping_old_to_new_block_ids: Optional[dict[int, int]] = None,
        mapping_real_to_desired_position_offset: Optional[dict[int, int]] = None,
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
            is_lazy=request.has_documents,
            q_offset=q_offset,
            q_mask=q_mask,
            need_to_copy=need_to_copy,
            mapping_old_to_new_block_ids=mapping_old_to_new_block_ids,
            mapping_real_to_desired_position_offset=mapping_real_to_desired_position_offset,
        )