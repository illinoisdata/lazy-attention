from dataclasses import dataclass, field
from itertools import chain
from typing import List

from drag.document import Document
from vllm.sequence import SequenceData
from vllm.transformers_utils.tokenizer import AnyTokenizer

@dataclass
class RAGSequence:
    sequence_id: int
    doc_ids: list[int]
    query_text: str
    query_token_ids: list[int]
    doc_token_ids: list[list[int]]


    _generated_token_ids:list[int] = field(init=False)
    _vllm_seq_data: SequenceData = field(init=False)

    def __post_init__(self):
        self._generated_token_ids = []
        # vllm_seq here means the sequence data structure of vllm, it is used for constructing decoding metadata for the QueryRequest.
        self._vllm_seq_data: SequenceData = SequenceData.from_seqs(
            # concatenate doc_token_ids and query_token_ids into a single list
            prompt_token_ids=list(chain.from_iterable(self.doc_token_ids)) + self.query_token_ids,
            output_token_ids=[]
        )

        # Read-only property for generated_token_ids
    @property
    def generated_token_ids(self) -> list[int]:
        return self._generated_token_ids

    # Read-only property for _vllm_seq_data
    @property
    def vllm_seq_data(self) -> SequenceData:
        return self._vllm_seq_data

    @classmethod
    def make_RAGSequence(cls, sequence_id: int, query: str, doc_ids: list[int], docDB: dict[int, Document], tokenizer: AnyTokenizer) -> "RAGSequence":
        query_token_ids = tokenizer.encode(query)
        doc_token_ids = [docDB[doc_id].token_ids for doc_id in doc_ids]
        return cls(sequence_id, doc_ids, query, query_token_ids, doc_token_ids)
    
    def append_generated_token(self, token_id: int, logprob: float) -> None:
        self._generated_token_ids.extend([token_id])
        self._vllm_seq_data.append_token_id(token_id, logprob)
        self._vllm_seq_data.update_num_computed_tokens(1)

    def finish_prefill(self) -> None:
        self._vllm_seq_data.update_num_computed_tokens(self._vllm_seq_data.get_prompt_len())

    # TODO: after preemption and resumption, we need to update the RAGSequence depending on how much info we still have in the memory.
    # scenerio 1: we still have the docs, prompt and part of the generated text in the memory, then update the _vllm_seq_data and _generated_token_ids
    # scenerio 2: even the prefilling part is partially lost, then directly overwrite the _vllm_seq_data and _generated_token_ids to new empty ones.