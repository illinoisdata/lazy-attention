
from typing import Optional, Any

from vllm import SamplingParams
from vllm.v1.request import Request



class Document:
    """A class to represent a document in the DynamicRAG system.
    """
    def __init__(self, raw_text: str, tokenizer: Any, 
                 block_size: int = 16, pad_token_id: int = 128001) -> None:
        self._raw_text = raw_text
        self.tokenizer = tokenizer
        self.alignment = block_size
        # TODO(haocheng): add tokenizer.pad_token_id, make it flexible
        self.pad_token_id = pad_token_id
        self._token_ids = None
        self._aligned_token_ids = None
        self._aligned_text = None
        self._hash_value = None
        
        self._align()
        self._hash()
        
    def _align(self):
        """Tokenize text and align to block_size for efficient KV-cache management."""
        self._token_ids = self.tokenizer.encode(self.raw_text, add_special_tokens=False)
        
        padding_length = (self.alignment - (len(self._token_ids) % self.alignment)) % self.alignment
        
        if padding_length > 0:
            pad_token_id = self.pad_token_id
            # self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
            self._aligned_token_ids = self._token_ids + [pad_token_id] * padding_length
        else:
            self._aligned_token_ids = self._token_ids.copy()
            
        # Decode the aligned token ids back to text
        # Note: This may not be necessary in production, but useful for debugging
        try:
            self._aligned_text = self.tokenizer.decode(self._aligned_token_ids)
        except Exception:
            self._aligned_text = None
    
    def _hash(self):
        """Generate a hash for the document, it is regarded as uid for doc.
        """
        if self._hash_value is None:
            import hashlib
            content_to_hash = self.raw_text
            if self.token_ids:
                content_to_hash += "_" + ",".join(map(str, self.token_ids[:10]))
            
            self._hash_value = hashlib.sha256(content_to_hash.encode()).hexdigest()
    
    @property
    def token_ids(self):
        return self._token_ids
    
    @property
    def aligned_token_ids(self):
        return self._aligned_token_ids
    
    @property
    def uid(self):
        return self._hash_value
    
    def __len__(self):
        return len(self._token_ids) if self._token_ids else 0
    
    def __str__(self):
        return self._raw_text
    
    def __repr__(self):
        return f"Document(length={len(self._aligned_text)}, hash={self._hash_value[:8]}...)"



class _Request(Request):
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
    ) -> None:
        super().__init__(request_id, prompt, prompt_token_ids, multi_modal_inputs, multi_modal_hashes, multi_modal_placeholders, sampling_params, eos_token_id, arrival_time, lora_request, structured_output_request)
        # extra attributes for DynamicRAG
        self.documents: Optional[list[Document]] = None
        # TODO(haocheng): add document keyword for hash
        # each document generate a hash value
        
        # then all token ids are concatenated