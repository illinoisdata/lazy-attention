import pytest
from vllm.distributed import cleanup_dist_env_and_memory
from minidrag.entrypoints import MiniDynamicRAG

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

@pytest.fixture
def vllm_reference_outputs(request, eager,
                           mock_model_name,
                           mock_sampling_params):
    value = request.config.cache.get(f"vllm_reference_outputs::{eager}", None)
    if not value:
        value = []
        MiniDynamicRAG.apply_triton_backend()
        import vllm
        llm = vllm.LLM(model=mock_model_name,
                       gpu_memory_utilization=0.9,
                       enforce_eager=eager,
                       enable_prefix_caching=True,)
        outputs = llm.generate(prompts, mock_sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            value.append((prompt, generated_text))
        request.config.cache.set(f"vllm_reference_outputs::{eager}", value)
        del llm
        cleanup_dist_env_and_memory()
    return value


class TestE2E:
    @pytest.mark.integration
    @pytest.mark.parametrize("eager", [True])
    def test_e2e(self, vllm_reference_outputs, eager,
                 mock_model_name,
                 mock_sampling_params):
        MiniDynamicRAG.apply_patches()
        import vllm
        llm = vllm.LLM(model=mock_model_name,                      
                       gpu_memory_utilization=0.9,
                       enforce_eager=eager,
                       enable_prefix_caching=True,)
        outputs = llm.generate(prompts, mock_sampling_params)
        print('-'*100)
        for output, pg_pair in zip(outputs, vllm_reference_outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            print(f"Reference generated text: {pg_pair[1]!r}")
        del llm
        cleanup_dist_env_and_memory()
