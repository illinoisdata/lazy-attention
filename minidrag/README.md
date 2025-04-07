# Minimal Dynamic RAG

This repo is a minimal implementation of Dynamic RAG.

We use monkeypatch to replace the original function/class in vLLM with our customized version.

For C/CUDA code, we use `#include` macro to inject the customized code into the original code.

## Organization

- `drag/`: the main implementation of Dynamic RAG.
- `drag/tests/`: the unit tests for Dynamic RAG.
- `drag/rotary_embedding.py`: the implementation of customized rotary embedding.
- `drag/llama.py`: the implementation of customized llama model.
- `drag/attention/backends/triton_attn.py`: the implementation of customized attention layer.


- `csrc/`: the customized CPP/CUDA code. We only inject a customized rotary embedding kernel into the original code. While the original code always rotate the key and query, we only rotate the query and keep the key unchanged.

# Build

```shell
module load cuda/12.4.0
conda install ccache
CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .
```


## Design

```mermaid
classDiagram
    LlamaAttention <|-- Attention
    LlamaAttention <|-- RotaryEmbedding
    Attention <|-- TritonAttentionImpl
    class LlamaAttention{
      +q,k,v
      +cos_sin_cache
      +rotary_dim
      +is_neox_style
      +forward()
    }
    class Attention{
      +attn_metadata
      +forward()
    }
    class RotaryEmbedding{
      +cos_sin_cache
      +rotary_dim
      +is_neox_style
      +forward_cuda() # rotate q only
    }
    class TritonAttentionImpl{
      +write_to_paged_cache()
      +chunked_prefill_paged_decode()
    }
```


## Appendix

AttentionMetadata is built in [gpu_model_runner.py](../vllm/v1/worker/gpu_model_runner.py).

```python
        attn_metadata = self.attn_metadata_builder.build(
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            common_prefix_len=common_prefix_len,
        )
```