import pytest

import torch._dynamo
torch._dynamo.config.suppress_errors = True

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# @pytest.fixture
# def vllm_reference_outputs(request):
#     value = request.config.cache.get("vllm_reference_outputs", None)
#     if not value:
#         value = []

#         from minidrag.attention.selector import apply_patch as apply_attn_selector_patch
#         from minidrag.attention.selector import revert_patch as revert_attn_selector_patch
#         from minidrag.platforms.cuda import apply_patch as apply_cuda_patch
#         from minidrag.platforms.cuda import revert_patch as revert_cuda_patch
#         apply_cuda_patch()
#         apply_attn_selector_patch()
#         import vllm
#         llm = vllm.LLM(model="meta-llama/Llama-3.2-1B",
#                        gpu_memory_utilization=0.9,
#                        enforce_eager=True,
#                        enable_prefix_caching=True,)
#         outputs = llm.generate(prompts, mock_sampling_params)

#         for output in outputs:
#             prompt = output.prompt
#             generated_text = output.outputs[0].text
#             print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#             value.append((prompt, generated_text))
#         request.config.cache.set("vllm_reference_outputs", value)
#         revert_attn_selector_patch()
#         revert_cuda_patch()
#     return value


@pytest.fixture(scope="module")
def mock_sampling_params():
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=100)


class TestE2E:
    @pytest.mark.integration
    def test_e2e(self, mock_sampling_params):
        # load patches
        from minidrag.platforms.cuda import apply_patch as apply_cuda_patch
        from minidrag.attention.layer import apply_patch as apply_attn_layer_patch
        from minidrag.attention.backends.triton_attn import apply_patch as apply_triton_attn_patch
        from minidrag.core.kv_cache_utils import apply_patch as apply_kv_cache_utils_patch
        from minidrag.model_executor.layers.rotary_embedding import apply_patch as apply_rotary_embedding_patch
        from minidrag.model_executor.models.llama import apply_patch as apply_llama_patch
        from minidrag._custom_ops import apply_patch as apply_custom_ops_patch

        # apply patches
        apply_cuda_patch()
        apply_attn_layer_patch()
        apply_triton_attn_patch()
        apply_kv_cache_utils_patch()
        apply_rotary_embedding_patch()
        apply_llama_patch()
        apply_custom_ops_patch()
        
        import vllm
        llm = vllm.LLM(model="meta-llama/Llama-3.2-1B",                      
                       gpu_memory_utilization=0.9,
                       enforce_eager=True,
                       enable_prefix_caching=True,)
        outputs = llm.generate(prompts, mock_sampling_params)
        print('-'*100)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
