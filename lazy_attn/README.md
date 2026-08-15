# `lazy` — LazyAttention over vLLM

The implementation. It is a **pure-Python monkey-patch layer** over vLLM plus
Triton kernels that are JIT-compiled at runtime: no C++/CUDA of its own, so the
prebuilt vLLM wheel is all it needs.

Full documentation lives in [`../docs`](../docs):

- **[design.md](../docs/design.md)** — the idea, the request lifecycle,
  per-document KV caching and hashing, the query-rotation metadata, the patch
  layer, and the limits.
- **[usage.md](../docs/usage.md)** — install, `document_seqs` requests offline and
  async, every environment switch, tests, troubleshooting.
- **[INTRO.md](./INTRO.md)** — short orientation on the two layers (kernels vs.
  patch registry) for someone about to change the code.

## Enable it

```python
import lazy.__vllm__   # patches vLLM at import time
from vllm import LLM   # now LazyLLM

llm.generate(prompts=[question], document_seqs=[[doc_a, doc_b, doc_c]])
```

## Layout

| path | role |
|---|---|
| `lazy/__vllm__.py`, `lazy/vllm_patch.py`, `lazy/ctxmgr.py` | the patch registry and its entry points |
| `lazy/entrypoints/`, `lazy/engine/` | frontend: `document_seqs` → tokenized, padded, hashed documents on the request |
| `lazy/request.py` | `LazyRequest`, and the synthetic per-document request |
| `lazy/core/` | scheduler (document readiness, merge, rotation metadata) and per-document KV cache lookups |
| `lazy/worker/` | model runner: metadata buffers and the packed block table |
| `lazy/model_executor/` | RoPE classes, Llama attention forward, in-kernel cos/sin |
| `lazy/attention/` | patched attention layer and backend; `ops/` = `lazy` variant, `old_ops/` = `mepic` variant |
| `lazy/utils/variants.py` | **every** runtime switch, in one documented place |
| `tests/` | unit and end-to-end tests |

## Develop

```bash
pytest tests -c tests/pytest.ini     # tests (see ../docs/usage.md §6)
make lint                           # flake8 + black + mypy
make fmt                            # isort + black
```

Install with [`../scripts/install.sh`](../scripts/install.sh), not `make install`:
vLLM has to be paired with a matching torch build, which a plain dependency
cannot express.

When changing behaviour, edit the concrete lazy implementation first; touch
`vllm_patch.py` only when the *set* of patched symbols changes.
