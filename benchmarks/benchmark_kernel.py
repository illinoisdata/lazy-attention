""" Here we directly benchmark forward pass of the model."""

from minidrag.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode as lazy_forward
from vllm.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode as origin_forward
from vllm.vllm_flash_attn import (flash_attn_varlen_func, get_scheduler_metadata)
