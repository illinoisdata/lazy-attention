# /////////////////////////////////////////////////////////////////////////////
# inject our custom ops
from .kv_cache_utils import hash_request_tokens_no_prefix
from .chunked_prefill_paged_decode import dynamic_chunked_prefill_paged_decode
from .model_executor.layers.rotary_embedding import dynamic_get_rope

# import original modules
import vllm.v1.core.kv_cache_utils
import vllm.attention.ops.chunked_prefill_paged_decode
import vllm.model_executor.layers.rotary_embedding

# # keep the original functions
# original_hash_request_tokens = vllm.v1.core.kv_cache_utils.hash_request_tokens
# original_chunked_prefill_paged_decode = vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode
# original_get_rope = vllm.model_executor.layers.rotary_embedding.get_rope

# # directly modify original module functions
# vllm.v1.core.kv_cache_utils.hash_request_tokens = hash_request_tokens_no_prefix
# vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode = dynamic_chunked_prefill_paged_decode
# vllm.model_executor.layers.rotary_embedding.get_rope = dynamic_get_rope

# /////////////////////////////////////////////////////////////////////////////
from vllm.envs import set_vllm_use_v1

# # make sure vllm uses v1
# set_vllm_use_v1(True)

# # /////////////////////////////////////////////////////////////////////////////
# """
# our custom ops are injected here, test the functionality by generation
# """
# run_vllm()

# # /////////////////////////////////////////////////////////////////////////////

# # restore the original functions
# vllm.v1.core.kv_cache_utils.hash_request_tokens = original_hash_request_tokens
# vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode = original_chunked_prefill_paged_decode
# vllm.model_executor.layers.rotary_embedding.get_rope = original_get_rope

# /////////////////////////////////////////////////////////////////////////////

class DynamicRAG:
    def __init__(self, config=None):
        # 保存原始函数
        self.original_functions = {
            'hash_request_tokens': vllm.v1.core.kv_cache_utils.hash_request_tokens,
            'chunked_prefill_paged_decode': vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode,
            'get_rope': vllm.model_executor.layers.rotary_embedding.get_rope
        }
        
        # 自定义实现
        self.custom_functions = {
            'hash_request_tokens': hash_request_tokens_no_prefix,
            'chunked_prefill_paged_decode': dynamic_chunked_prefill_paged_decode,
            'get_rope': dynamic_get_rope
        }
        
        # 配置
        self.config = config or {}
        self.use_v1 = True
        
    def __enter__(self):
        """进入上下文时替换函数"""
        # 设置vllm使用v1
        set_vllm_use_v1(self.use_v1)
        
        # 替换函数
        vllm.v1.core.kv_cache_utils.hash_request_tokens = self.custom_functions['hash_request_tokens']
        vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode = self.custom_functions['chunked_prefill_paged_decode']
        vllm.model_executor.layers.rotary_embedding.get_rope = self.custom_functions['get_rope']
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时恢复原始函数"""
        # 恢复原始函数
        vllm.v1.core.kv_cache_utils.hash_request_tokens = self.original_functions['hash_request_tokens']
        vllm.attention.ops.chunked_prefill_paged_decode.chunked_prefill_paged_decode = self.original_functions['chunked_prefill_paged_decode']
        vllm.model_executor.layers.rotary_embedding.get_rope = self.original_functions['get_rope']
        
    def run(self, *args, **kwargs):
        """运行vllm"""
        with self:
            return run_vllm(*args, **kwargs)
            
    def configure(self, **kwargs):
        """更新配置"""
        self.config.update(kwargs)
        if 'use_v1' in kwargs:
            self.use_v1 = kwargs['use_v1']
            
    def get_config(self):
        """获取当前配置"""
        return self.config.copy()