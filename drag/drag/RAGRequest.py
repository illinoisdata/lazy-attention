from dataclasses import dataclass
from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from drag.document import Document

class RAGRequestPhase(Enum):
    PREFILL = 0
    DECODE = 1

class RAGRequestType(Enum):
    CACHE_DOC = 0
    QUERY = 1

# Abstract Base Class
@dataclass
class RAGRequest(ABC):
    request_id: int
    original_seq_id: int

    @abstractmethod
    def get_type(self) -> RAGRequestType:
        """Abstract method to get request type"""
        pass

# Subclass for CACHE_DOC requests
@dataclass
class CacheDocRequest(RAGRequest):
    doc_id: int
    doc_token_ids: List[int]
    doc_length: int
    
    @classmethod
    def from_document(cls, request_id: int, doc_id: int, doc: Document, original_seq_id: int):
        return cls(
            request_id=request_id,
            original_seq_id=original_seq_id,
            doc_id=doc_id,
            doc_token_ids=doc.token_ids,
            doc_length=len(doc.token_ids)
        )

    def get_type(self) -> RAGRequestType:
        return RAGRequestType.CACHE_DOC

# Subclass for QUERY requests
@dataclass
class QueryRequest(RAGRequest):
    prompt_ids: List[int]
    doc_ids: List[int]

    def get_type(self) -> RAGRequestType:
        return RAGRequestType.QUERY




