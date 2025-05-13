""" In flash attention metadata, we add a new field for our lazy attention.

The structure of the metadata is as follows:
[[doc_len, offset],
 [doc_len, offset],
 [doc_len, offset],
 ...
 [doc_len, offset]]

Where doc_len is the length of the document and offset is for query rotation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.layer import Attention
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import (flash_attn_supports_fp8,
                                           get_flash_attn_version)
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if current_platform.is_cuda():
    from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                      get_scheduler_metadata)

@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |--------------------[] seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    
    # For lazy attention.
    # Each query can has document(s) optionally. When the number of 
    # documents is 0, it means that the query is not a document query. 
    nums_docs: torch.Tensor
    lens_docs: torch.Tensor  
    doc_padding_lens: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None

    # for local attention
    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor
        local_seqused_k: torch.Tensor
        local_block_table: torch.Tensor
        local_max_query_len: int
        local_max_seq_len: int
        local_scheduler_metadata: Optional[torch.Tensor]

    local_attn_metadata: Optional[LocalAttentionMetadata] = None