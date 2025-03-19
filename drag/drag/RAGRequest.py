from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from drag.document import Document

class RAGRequestPhase(Enum):
    PREFILL = 0
    DECODE = 1

# Abstract Base Class
class RAGRequest(ABC):
    def __init__(self, request_id: int, original_seq_id: int):
        self.request_id = request_id
        self.original_seq_id = original_seq_id

    @abstractmethod
    def get_type(self) -> str:
        """Abstract method to get request type"""
        pass

# Subclass for CACHE_DOC requests
class CacheDocRequest(RAGRequest):
    def __init__(self, request_id: int, doc_id:int, doc: Document, original_seq_id: int):
        super().__init__(request_id, original_seq_id)
        self.doc_id:int = doc_id
        self.doc_token_ids:List[int] = doc.token_ids
        self.doc_length = len(doc.token_ids)

    def get_type(self) -> str:
        return "CACHE_DOC"

# Subclass for QUERY requests
class QueryRequest(RAGRequest):
    def __init__(self, request_id: int, prompt: str, doc_ids: List[int], original_seq_id:int):
        super().__init__(request_id)
        self.prompt_ids:list[int] = prompt
        self.doc_ids:List[int] = doc_ids

    def get_type(self) -> str:
        return "QUERY"
    
  


