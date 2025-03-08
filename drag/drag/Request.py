from dataclasses import dataclass
from enum import Enum
from typing import List

class RequestType(Enum):
    CACHE_DOC = 0
    QUERY = 1

@dataclass
class Request:
    request_id: int
    prompt: str
    document_ids: list[int]
    type: RequestType
   

