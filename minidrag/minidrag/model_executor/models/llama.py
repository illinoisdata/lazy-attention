import torch


# class LlamaAttention(nn.Module):
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # print("Use customized llama attn forward function")
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # customized rotary embedding, only rotate query but keep key unchanged
    # keep the signature of the original rotary embedding
    q, k = self.rotary_emb(positions, q, k)
        
    attn_output = self.attn(q, k, v,
                            cos_sin_cache=self.rotary_emb.cos_sin_cache, 
                            rotary_dim=self.rotary_emb.rotary_dim,
                            is_neox_style=self.rotary_emb.is_neox_style)
    output, _ = self.o_proj(attn_output)
    return output


# original forward function
forward_original = forward

def apply_patch():
    global forward_original
    import vllm.model_executor.models.llama
    forward_original = vllm.model_executor.models.llama.LlamaAttention.forward
    vllm.model_executor.models.llama.LlamaAttention.forward = forward

def revert_patch():
    import vllm.model_executor.models.llama
    vllm.model_executor.models.llama.LlamaAttention.forward = forward_original