import pytest
import torch

from vllm.distributed import cleanup_dist_env_and_memory
from minidrag.entrypoints import MiniDynamicRAG

from utils import set_seed


# -----------------------------------------------------------------------------
# TritonAttn V1 Backend
# -----------------------------------------------------------------------------
# Note(haocheng): make sure only one model is loaded at a time
@pytest.fixture(scope="module")
def vllm_ref_outputs_simple(mock_prompts,
                            mock_model_name,
                            mock_sampling_params):
    torch.cuda.empty_cache()
    llm = None
    try:
        MiniDynamicRAG.apply_triton_backend()
        value = []
        set_seed(42)
        import vllm
        llm = vllm.LLM(model=mock_model_name,
                        gpu_memory_utilization=0.9,
                        enforce_eager=True,
                        enable_prefix_caching=True,
                        seed=42,)
        outputs = llm.generate(mock_prompts, mock_sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_token_ids = output.outputs[0].token_ids
            value.append((prompt, generated_text, generated_token_ids))
        return value
    finally:
        if llm is not None:
            del llm
        MiniDynamicRAG.revert_triton_backend()
        cleanup_dist_env_and_memory()
        torch.cuda.synchronize()
        


class TestE2E:
    @pytest.mark.gpu
    @pytest.mark.integration
    @pytest.mark.parametrize("eager", [True, False])
    def test_e2e_simple(self, 
                        eager,
                        mock_prompts,
                        mock_model_name,
                        mock_sampling_params,
                        vllm_ref_outputs_simple,):
        ref_outputs = vllm_ref_outputs_simple
        torch.cuda.empty_cache()
        llm = None
        try:
            with MiniDynamicRAG():
                set_seed(42)
                import vllm
                llm = vllm.LLM(model=mock_model_name,                      
                            gpu_memory_utilization=0.9,
                            enforce_eager=eager,
                            enable_prefix_caching=True,
                            seed=42,)
                outputs = llm.generate(mock_prompts, mock_sampling_params)
                for output, pgt_tuple in zip(outputs, ref_outputs):
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                    print(f"Reference generated text: {pgt_tuple[1]!r}")
                    assert prompt == pgt_tuple[0]
                    # TODO(haocheng): the generated text is not exactly the same 
                    # as the length increases
                    num_tokens = min(10, len(pgt_tuple[2]))
                    assert output.outputs[0].token_ids[:num_tokens] == \
                           pgt_tuple[2][:num_tokens], \
                        f"Token IDs mismatch:\n"\
                        f"gen: {output.outputs[0].token_ids[:num_tokens]}\n" \
                        f"ref: {pgt_tuple[2][:num_tokens]}"
        finally:
            if llm is not None:
                del llm
            cleanup_dist_env_and_memory()
            torch.cuda.synchronize()
            
    # @pytest.mark.gpu
    # @pytest.mark.integration
    # @pytest.mark.parametrize("eager", [False, True])
    # def test_e2e_reorder(self,
    #                      eager,
    #                      mock_model_name,
    #                      mock_sampling_params,
    #                      vllm_ref_outputs_simple,):
    #     pass