
from typing import Optional

from vllm import SamplingParams
from vllm.v1.request import Request
from vllm.v1.structured_output.request import StructuredOutputRequest

from minidrag.engine.__init__ import EngineCoreRequest


class WrapperedRequest(Request):
    """Extends the Request class to support DynamicRAG.
    """
    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        multi_modal_inputs= None,
        multi_modal_hashes = None,
        multi_modal_placeholders = None,
        lora_request = None,
        structured_output_request = None,
        documents_token_ids: Optional[list[list[int]]] = None,
        documents_hash: Optional[list[str]] = None,
    ) -> None:
        super().__init__(request_id, prompt, prompt_token_ids, 
                         multi_modal_inputs, multi_modal_hashes, multi_modal_placeholders, 
                         sampling_params, eos_token_id, arrival_time, 
                         lora_request, structured_output_request)
        # extra attributes for DynamicRAG
        self.documents_token_ids = documents_token_ids
        self.documents_hash = documents_hash
        self.len_documents = None
        self.num_computed_tokens_documents = None
        if documents_token_ids is not None:
            # assert len(documents_token_ids) == len(documents_hash)
            self.len_documents = len(documents_token_ids)
            self.num_computed_tokens_documents = [0 for _ in range(len(documents_hash))]
            
        
    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
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
            documents_token_ids=request.documents_token_ids,
            documents_hash=request.documents_hash,
        )
    
    def __repr__(self) -> str:
        return f"WrapperedRequest(request_id={self.request_id}, prompt={self.prompt}, " \
               f"prompt_token_ids={self.prompt_token_ids}, sampling_params={self.sampling_params}, " \
               f"eos_token_id={self.eos_token_id}, arrival_time={self.arrival_time}, " \
               f"documents_token_ids={self.documents_token_ids}, documents_hash={self.documents_hash}), " \
               f"len_documents={self.len_documents})" \
               f"num_computed_tokens_documents={self.num_computed_tokens_documents})"
               

def apply_patch():
    """Apply the patch to the Request class.
    """
    import vllm.v1.request
    vllm.v1.request.Request = WrapperedRequest