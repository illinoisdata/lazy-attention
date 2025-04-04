# /////////////////////////////////////////////////////////////////////////////
# inject our custom ops
from .kv_cache_utils import hash_request_tokens_no_prefix
from .chunked_prefill_paged_decode import dynamic_chunked_prefill_paged_decode
from .rotary_embedding import dynamic_get_rope

# import original modules
import vllm.v1.core.kv_cache_utils
import vllm.attention.ops.chunked_prefill_paged_decode
import vllm.model_executor.layers.rotary_embedding

# keep the original functions
original_hash_request_tokens = vllm.v1.core.kv_cache_utils.hash_request_tokens
original_chunked_prefill_paged_decode = vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode
original_get_rope = vllm.model_executor.layers.rotary_embedding.get_rope

# directly modify original module functions
vllm.v1.core.kv_cache_utils.hash_request_tokens = hash_request_tokens_no_prefix
vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode = dynamic_chunked_prefill_paged_decode
vllm.model_executor.layers.rotary_embedding.get_rope = dynamic_get_rope

# /////////////////////////////////////////////////////////////////////////////
from vllm.envs import set_vllm_use_v1

# make sure vllm uses v1
set_vllm_use_v1(True)

# /////////////////////////////////////////////////////////////////////////////
"""
our custom ops are injected here, test the functionality by generation
"""
run_vllm()

# /////////////////////////////////////////////////////////////////////////////

# restore the original functions
vllm.v1.core.kv_cache_utils.hash_request_tokens = original_hash_request_tokens
vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode = original_chunked_prefill_paged_decode
vllm.model_executor.layers.rotary_embedding.get_rope = original_get_rope