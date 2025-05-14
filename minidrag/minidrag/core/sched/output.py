# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata)
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
    from vllm.sampling_params import SamplingParams
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
    is_lazy_req: bool = False
    lazy_mask: Optional[list[int]] = None
    lazy_offset: Optional[list[int]] = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: list[int],
        lazy_mask: list[int],
        lazy_offset: list[int],
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
            is_lazy_req = request.has_documents,
            lazy_mask = lazy_mask,
            lazy_offset = lazy_offset,
        )

    def __repr__(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids={self.prompt_token_ids},"
                f"mm_inputs={self.mm_inputs},"
                f"mm_hashes={self.mm_hashes},"
                f"mm_positions={self.mm_positions},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request,}"
                f"is_lazy_req={self.is_lazy_req},"
                f"lazy_mask={self.lazy_mask},"
                f"lazy_offset={self.lazy_offset}"
                ")")

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids_len={len(self.prompt_token_ids)},"
                f"mm_inputs={self.mm_inputs},"
                f"mm_hashes={self.mm_hashes},"
                f"mm_positions={self.mm_positions},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request}"
                ")")