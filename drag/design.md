# Design

```mermaid
sequenceDiagram
    participant a as paged_attn.py
    participant b as prefix_prefill.py
    participant c as _custom_ops.py
    a->>b: forward_prefix -- context_attention_fwd
    a->>c: forward_decode -- dynamic_paged_attention
    b-->>a: processed query prompt (first token)
    c-->>a: generate new following tokens

    loop Gen
        c->>c: generation
    end
```

### Prefill

When device capability >= 80, block size is 128, otherwise 64.

When fp32, reduce block size due to limited GPU Shared memory.

Then we compute `grid`, i.e., the number of kernel instances that run in parallel.
