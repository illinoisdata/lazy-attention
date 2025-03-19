from dataclasses import dataclass

@dataclass
class RAGSequence:
    sequence_id: int
    doc_ids: list[int]
    query_token_ids: list[int]
    doc_token_ids: list[list[int]]
    generated_token_ids: list[int]
    seq_token_ids: list[int] #union of doc_ids, query_token_ids, and generated_token_ids(if any)
