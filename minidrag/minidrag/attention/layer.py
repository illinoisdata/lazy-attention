"""
Attention forward function.

Accepts cos_sin_cache, rotary_dim, and is_neox_style as arguments for custom 
rotary embedding operations.
"""

import torch
from typing import Optional
from vllm.forward_context import get_forward_context, ForwardContext
from vllm.utils import direct_register_custom_op
from vllm.platforms import current_platform
from vllm.attention.layer import maybe_save_kv_layer_to_connector, wait_for_kv_layer_from_connector

# class Attention(nn.Module):
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor = None,
    rotary_dim: int = None,
    is_neox_style: bool = False,
    # For some alternate attention backends like MLA the attention output
    # shape does not match the query shape, so we optionally let the model
    # definition specify the output tensor shape.
    output_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(query, key, value)
        if self.use_output:
            output_shape = (output_shape
                            if output_shape is not None else query.shape)
            output = torch.empty(output_shape,
                                 dtype=query.dtype,
                                 device=query.device)
            hidden_size = output_shape[-1]
            # We skip reshaping query, key and value tensors for the MLA
            # backend since these tensors have different semantics and are
            # processed differently.
            if not self.use_mla:
                # Reshape the query, key, and value tensors.
                # NOTE(woosuk): We do this outside the custom op to minimize the
                # CPU overheads from the non-CUDA-graph regions.
                query = query.view(-1, self.num_heads, self.head_size)
                output = output.view(-1, self.num_heads, self.head_size)
                if key is not None:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                if value is not None:
                    value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                lazy_metadata = forward_context.lazy_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(self,
                                  query,
                                  key,
                                  value,
                                  self_kv_cache,
                                  attn_metadata,
                                  lazy_metadata,
                                  cos_sin_cache,
                                  rotary_dim,
                                  is_neox_style,
                                  output=output)
            else:
                # We use dynamic_unified_attention_with_output to replace
                # unified_attention_with_output to support dynamic RAG's block attention.
                torch.ops.vllm.dynamic_unified_attention_with_output(
                    query, key, value, output, self.layer_name,
                    cos_sin_cache, rotary_dim, is_neox_style)
            return output.view(-1, hidden_size)
        else:
            raise ValueError(
                "Attention layer must be configured to use output tensor. ")
            
            
def dynamic_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    lazy_metadata = forward_context.lazy_metadata
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      lazy_metadata,
                      cos_sin_cache,
                      rotary_dim,
                      is_neox_style,
                      output=output)
    
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)


def dynamic_unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    return

direct_register_custom_op(
    op_name="dynamic_unified_attention_with_output",
    op_func=dynamic_unified_attention_with_output,
    mutates_args=["output"],
    fake_impl=dynamic_unified_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
)


# class CompilationConfig(BaseModel):
def set_splitting_ops_for_v1(self):
    # If default, override splitting ops for piecewise cudagraph on V1.
    # NOTE: this function needs to be called
    if not self.splitting_ops:
        self.splitting_ops = [
            "vllm.unified_attention",
            "vllm.unified_attention_with_output",
            "vllm.dynamic_unified_attention_with_output",
        ]


original_forward = None
original_splitting_ops = None


def apply_patch():
    import vllm.attention.layer
    import vllm.config
    global original_forward, original_splitting_ops

    original_forward = vllm.attention.layer.Attention.forward
    vllm.attention.layer.Attention.forward = forward

    original_splitting_ops = vllm.config.CompilationConfig.set_splitting_ops_for_v1
    vllm.config.CompilationConfig.set_splitting_ops_for_v1 = set_splitting_ops_for_v1

def revert_patch():
    import vllm.attention.layer
    import vllm.config
    vllm.attention.layer.Attention.forward = original_forward
    vllm.config.CompilationConfig.set_splitting_ops_for_v1 = original_splitting_ops