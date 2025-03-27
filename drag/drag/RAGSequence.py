from dataclasses import dataclass
from typing import List

from drag.document import Document
from vllm.transformers_utils.tokenizer import AnyTokenizer

@dataclass
class RAGSequence:
    sequence_id: int
    doc_ids: list[int]
    query_text: str
    query_token_ids: list[int]
    doc_token_ids: list[list[int]]
    generated_token_ids: list[int]

    @classmethod
    def make_RAGSequence(cls, sequence_id: int, query: str, doc_ids: list[int], docDB: dict[int, Document], tokenizer: AnyTokenizer) -> "RAGSequence":
        query_token_ids = tokenizer.encode(query)
        doc_token_ids = [docDB[doc_id].token_ids for doc_id in doc_ids]
        return cls(sequence_id, doc_ids, query, query_token_ids, doc_token_ids, [])