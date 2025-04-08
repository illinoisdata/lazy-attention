import pytest
import torch

@pytest.fixture
def mock_attention_input():
    # mock positions and hidden_states
    positions = torch.tensor([0])
    hidden_states = torch.tensor([[1, 2, 3]])
    return positions, hidden_states

class TestLlama:
    def test_llama_attention(self, mock_attention_input):
        # get the original LlamaAttention
        from vllm.model_executor.models.llama import LlamaAttention
        ref_llama_attention = LlamaAttention(
            num_heads=1,
            head_size=1,
            scale=1,
            num_kv_heads=1,
            use_direct_call=True,
        )
        ref_output = ref_llama_attention.forward(
            positions=mock_attention_input[0],
            hidden_states=mock_attention_input[1],
        )

        from minidrag.model_executor.models.llama import apply_patch 
        
        llama_attention = LlamaAttention(
            num_heads=1,
            head_size=1,
            scale=1,
            num_kv_heads=1,
            use_direct_call=True,
        )
        output = llama_attention.forward(
            positions=mock_attention_input[0],
            hidden_states=mock_attention_input[1],
        )
        assert output.shape == ref_output.shape
        assert torch.allclose(output, ref_output, atol=1e-5)
