"""
Sync RAG entrypoints

Here we offer sync RAG to serving batched requests.

We support:

- [ ] LazyAttention
- [ ] vLLM no cache
- [ ] vLLM prefix caching


Note: for this experiments, we use `LLM.generate()` for batched inference.

TTFT is from `output.metrics.first_token_time - output.metrics.first_scheduled_time`
to align with the setting in CacheBlend.
"""

from __future__ import annotations

# ------------------------------------------------------------
# Deal with LazyAttention monkey patch
# ------------------------------------------------------------
import os
if os.environ.get("VLLM_USE_LAZY_ATTENTION", "0") == "1":
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
    import lazy.__vllm__
# ------------------------------------------------------------

from typing import Dict, Any

from abc import ABC, abstractmethod


class SyncRAG(ABC):
    @abstractmethod
    def cache_docs(self, docs: list[str]) -> None:
        """
        For each method, we need to prepare the KV cache before evaluation.
        """
        pass

    @abstractmethod
    def generate(self, query: str, num_docs: int) -> str:
        pass

    @abstractmethod
    def free_docs(self) -> None:
        """
        Free the KV cache of docs if needed.
        """
        pass

    def get_stats_dict(self) -> Dict[str, Any]:
        return {}
