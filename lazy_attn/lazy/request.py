"""
Request class for LazyAttention.

This class is a wrapper around the original Request class from vllm. 
"""

import copy
import enum
from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.v1.engine import (EngineCoreEvent,
                            FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest


from lazy.engine.__init__ import EngineCoreRequest, EngineCoreEventType



class LazyRequest:

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        multi_modal_inputs: Optional[list[MultiModalKwargs]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: Optional[SamplingParams],
        eos_token_id: Optional[int],
        arrival_time: float,
        pooling_params: Optional[PoolingParams] = None,
        client_index: int = 0,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
        priority: int = 0,
        # Extra attributes for DynamicRAG
        documents_token_ids_padded: Optional[list[list[int]]] = None,
        document_lens: Optional[list[int]] = None,
        document_lens_padded: Optional[list[int]] = None,
        document_seq_hash: Optional[str] = None,
        is_document_request: bool = False,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request

        self.status = RequestStatus.WAITING
        if sampling_params and sampling_params.guided_decoding is not None:
            self.status = RequestStatus.WAITING_FOR_FSM
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: Optional[dict[str, Any]] = None

        if pooling_params is not None:
            self.max_tokens = 1
        elif sampling_params is not None:
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if sampling_params.extra_args is not None:
                self.kv_transfer_params = \
                    sampling_params.extra_args.get("kv_transfer_params")
        else:
            raise ValueError(
                "sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: Optional[str] = cache_salt

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: list[str] = multi_modal_hashes or []
        self.num_encoder_inputs = len(self.mm_inputs)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

        # Read-only views
        # Prevent directly appending to the these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # /////////////////////////////////////////
        self.arrival_time = arrival_time
        # Get extra attributes for LazyAttention
        self.is_document_request = is_document_request
        self.documents_token_ids_padded = documents_token_ids_padded
        self.document_lens = document_lens
        self.document_lens_padded = document_lens_padded
        self.document_seq_hash = document_seq_hash
        self.num_computed_tokens_docs = ([0 for _ in document_lens]
                                         if document_lens is not None else None)
        # Set by merge_documents(). A request can be scheduled out of the
        # waiting queue more than once -- preemption puts it back -- so the
        # merge has to know whether it already happened.
        self.documents_merged = False

    def merge_documents(self) -> bool:
        """Prepend the document token ids to the prompt. Idempotent.

        Returns True if this call did the merge, False if it was already done.

        Idempotence is not cosmetic. A preempted request re-enters the waiting
        queue and is scheduled again, so without the guard the documents were
        prepended a second time -- the prompt grew by another full copy of
        them, `num_computed_tokens` then exceeded `num_tokens`, and the
        scheduler died on `assert num_new_tokens > 0`.
        """
        from itertools import chain
        assert self.has_documents
        if self.documents_merged:
            return False
        self.prompt_token_ids = (list(chain.from_iterable(self.documents_token_ids_padded)) +
                                 self.prompt_token_ids)
        self.num_prompt_tokens = len(self.prompt_token_ids)
        # Tokens generated so far are kept. Rebuilding _all_token_ids from the
        # prompt alone discarded them, which is only invisible while this runs
        # exactly once, before the request has generated anything.
        self._all_token_ids = self.prompt_token_ids + self._output_token_ids
        self.all_token_ids = ConstantList(self._all_token_ids)
        self.documents_merged = True
        return True

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "LazyRequest":
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(request.mm_inputs, MultiModalKwargs), (
                "mm_inputs was not updated in EngineCore.add_request")

        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params)
            if request.sampling_params else None,
            cache_salt=request.cache_salt,
            priority=request.priority,
            # Extra attributes for LazyAttention
            documents_token_ids_padded=request.documents_token_ids_padded,
            document_seq_hash=request.document_seq_hash,
            document_lens=request.document_lens,
            document_lens_padded=request.document_lens_padded,
        )

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
            
    @property
    def is_output_corrupted(self) -> bool:
        return self.num_nans_in_logits > 0

    @property
    def has_documents(self) -> bool:
        return self.documents_token_ids_padded is not None

    def document_request(self, doc_idx: int) -> "LazyRequest":
        """The request that populates one document's KV blocks.

        It only ever prefills. The blocks it writes are the ones this request
        later looks up, so both sides hash the *same object* -- salt, and
        anything vLLM adds to the block hash later, line up by construction
        rather than by two code paths agreeing.

        `lora_request` is deliberately not forwarded, and because both the
        lookup and the write go through this method, the two sides still agree
        -- the parent would see its documents as ready and proceed on document
        KV computed by the base model while its own query ran under the
        adapter. That silent mix is why documents + LoRA is rejected up front
        in `LazyProcessor` rather than handled here.
        """
        assert self.has_documents
        sampling_params = copy.deepcopy(self.sampling_params)
        sampling_params.max_tokens = 1  # TODO(haocheng): how to avoid
        return LazyRequest(
            request_id=f"{self.request_id}_d{doc_idx}",
            prompt_token_ids=self.documents_token_ids_padded[doc_idx],
            multi_modal_inputs=self.mm_inputs,
            multi_modal_hashes=self.mm_hashes,
            multi_modal_placeholders=self.mm_positions,
            sampling_params=sampling_params,
            eos_token_id=self.eos_token_id,
            arrival_time=self.arrival_time,
            cache_salt=self.cache_salt,
            is_document_request=True,
        )

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id].length
        return num_tokens

    @property
    def use_structured_output(self) -> bool:
        # `sampling_params` is None for pooling requests (embedding /
        # classification), which the scheduler still asks about -- so the
        # None guard is load-bearing, not defensive.
        return (self.sampling_params is not None
                and self.sampling_params.guided_decoding is not None)

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: Optional[float] = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __repr__(self) -> str:
        detailed = False
        if not detailed:
            return f"Request(request_id={self.request_id})"

        return f"Request(request_id={self.request_id}," \
               f"prompt_token_ids={self.prompt_token_ids}, sampling_params={self.sampling_params}, " \
               f"eos_token_id={self.eos_token_id}, arrival_time={self.arrival_time}, " \
               f"documents_token_ids_padded={self.documents_token_ids_padded}," \
               f"document_lens={self.document_lens}," \
               f"document_lens_padded={self.document_lens_padded}," \
               f"num_computed_tokens_docs={self.num_computed_tokens_docs})"


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    # For lazy attention with documents
    DOC_WAITING = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}

def apply_patch():
    """Apply the patch to the Request class.
    """
    import vllm.v1.request
    vllm.v1.request.Request = LazyRequest
