# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Optional

from vllm.v1.core.sched.output import NewRequestData as BaseNewRequestData

if TYPE_CHECKING:
    from lazy.request import LazyRequest as Request  # Replaces the V1 Request


@dataclass
class NewRequestData(BaseNewRequestData):
    """vLLM's NewRequestData plus the per-request lazy rotation metadata.

    Subclassing (rather than forking the dataclass) means fields added upstream
    -- pooling_params and the tuple-of-groups block_ids both landed in 0.9.x --
    are inherited instead of needing to be re-copied on every vLLM bump.
    """

    # Lazy attention, if one req has documents, it is lazy
    is_lazy: bool = False  # [num_seqs]
    lazy_variant: int = 0
    q_offset: Optional[list[int]] = None  # [num_seqs, num_blocks]
    q_mask: Optional[list[int]] = None  # [num_seqs, num_blocks]

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        lazy_variant: int = 0,
        q_offset: Optional[list[int]] = None,
        q_mask: Optional[list[int]] = None,
    ) -> NewRequestData:
        base = BaseNewRequestData.from_request(request, block_ids)
        base_kwargs = {f.name: getattr(base, f.name) for f in fields(base)}
        return cls(
            **base_kwargs,
            is_lazy=request.has_documents,
            lazy_variant=lazy_variant,
            q_offset=q_offset,
            q_mask=q_mask,
        )
