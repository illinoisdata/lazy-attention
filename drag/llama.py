import torch

from vllm.modeling.attention import AttentionMetadata

# class LlamaAttention(nn.Module):
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # customized rotary embedding, only for query
    q, cos_sin_cache, rotary_dim = self.rotary_emb(positions, q)
        
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                            cos_sin_cache=cos_sin_cache, 
                            rotary_dim=rotary_dim,
                            is_neox_style=self.rotary_emb.is_neox_style)
    output, _ = self.o_proj(attn_output)
    return output