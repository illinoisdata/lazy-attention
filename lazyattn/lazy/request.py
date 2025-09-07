"""
Request class for LazyAttention.

This class is a wrapper around the original Request class from vllm. 
"""

import enum
from typing import TYPE_CHECKING, Optional, Union

from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
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
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        # Extra attributes for DynamicRAG
        documents_token_ids_padded: Optional[list[list[int]]] = None,
        document_lens: Optional[list[int]] = None,
        document_lens_padded: Optional[list[int]] = None,
        document_seq_hash: Optional[str] = None,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request

        self.status = (RequestStatus.WAITING_FOR_FSM
                       if sampling_params.guided_decoding is not None else
                       RequestStatus.WAITING)
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0

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

        # /////////////////////////////////////////
        self.arrival_time = arrival_time
        # Get extra attributes for LazyAttention
        self.is_document_request = False
        self.documents_token_ids_padded = documents_token_ids_padded
        self.document_lens = document_lens
        self.document_lens_padded = document_lens_padded
        self.document_seq_hash = document_seq_hash
        self.num_computed_tokens_docs = ([0 for _ in document_lens] 
                                         if document_lens is not None else None)
            
    def merge_documents(self):
        """Merge the document token ids into the prompt token ids."""
        from itertools import chain
        assert self.has_documents
        self.prompt_token_ids = (list(chain.from_iterable(self.documents_token_ids_padded)) + 
                                 self.prompt_token_ids)
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._all_token_ids = self.prompt_token_ids.copy()
        self.all_token_ids = ConstantList(self._all_token_ids)

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "LazyRequest":
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(request.mm_inputs, MultiModalKwargs), (
                "mm_inputs was not updated in EngineCore.add_request")

        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params),
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
    def has_documents(self) -> bool:
        return self.documents_token_ids_padded is not None

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
        return self.sampling_params.guided_decoding is not None

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
