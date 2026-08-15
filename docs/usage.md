# LazyAttention — Usage

How to install LazyAttention, send it requests, and tune it. For why it works the
way it does, see [design.md](./design.md).

---

## 1. Install

```bash
bash scripts/install.sh --venv .venv   # ~2 minutes
source .venv/bin/activate
python scripts/validate_lazy.py        # end-to-end check
```

LazyAttention adds no C++/CUDA of its own — it is Python monkey patches over vLLM
plus Triton kernels JIT-compiled at runtime — so the prebuilt vLLM wheel already
contains every compiled kernel it needs. The installer pins the combination that
works (vLLM 0.9.2 / torch 2.7.0+cu128 / transformers 4.53.2, and Triton ≥ 3.4 on
`sm_120`); see the [root README](../README.md#install) for why each pin exists.

Useful flags: `--check` reports on the environment without installing, `--bench`
also installs the benchmark dependencies, `--source` builds vLLM from source.

Hardware: `sm_80` (A100) through `sm_120` (RTX 50-series).

## 2. Enabling LazyAttention

One import, before anything touches vLLM:

```python
import lazy.__vllm__   # noqa: F401  — patches vLLM at import time
from vllm import LLM, SamplingParams
```

That call patches `vllm.LLM`, the processor, engine, scheduler, model runner and
the attention stack in place, and forces
`VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1`. After it, plain `vllm.LLM` *is*
`LazyLLM` — importing `lazy.entrypoints.llm.LazyLLM` explicitly works too and is
equivalent.

`LazyAttentionContextManager` is **not** a scoped version of this. It applies
the attention-path patches only — the RoPE classes, the attention layer and the
Triton backend — and arranges for the engine-core subprocess to do the same. It
leaves `vllm.LLM`, the processor and the scheduler alone, so `document_seqs`
inside the block would be an unexpected keyword argument:

```python
from lazy.ctxmgr import LazyAttentionContextManager

with LazyAttentionContextManager():
    ...   # lazy kernels, stock frontend
```

Use it to get the lazy kernels into a process (kernel tests, profiling); use
`import lazy.__vllm__` for anything that sends documents.

## 3. Offline inference

Pass the documents **beside** the prompt, not inside it:

```python
import lazy.__vllm__  # noqa: F401
from vllm import LLM, SamplingParams

llm = LLM(model="hxia7/Llama-3.2-1B-Block-FT", max_model_len=2048)
params = SamplingParams(temperature=0, max_tokens=64)

docs = [
    "Paris is the capital and largest city of France.",
    "Berlin is the capital of Germany.",
    "Rome is the capital of Italy.",
]

out = llm.generate(
    prompts=["Question: Which city is the capital of France? Answer:"],
    sampling_params=params,
    document_seqs=[docs],          # one list of documents per prompt
)
print(out[0].outputs[0].text)
```

`document_seqs` is a list parallel to `prompts`: element *i* is the list of
documents for prompt *i*. Omit it (or pass `None`) and the request is handed
straight to upstream vLLM — including features the document path cannot offer,
such as parallel sampling (`n > 1`).

### What the engine does with it

Each document is tokenized on its own, has its BOS dropped, is right-padded to a
multiple of `block_size`, and gets its own KV blocks. Documents that are already
cached are reused; missing ones are prefilled as their own requests first. Then
the documents are spliced in front of the prompt and generation proceeds normally.

### Prompt construction rules

* **Documents are prepended, in order.** The effective prompt is
  `doc[0] + doc[1] + … + prompt`. Do **not** also inline the documents in
  `prompts` — they would appear twice.
* **A preamble that must precede the documents belongs in `document_seqs[0]`.**
  There is no slot before the document region.
* **The prompt holds everything after the documents**: the question, the
  instruction, and any chat-template suffix (e.g. the
  `<|eot_id|><|start_header_id|>assistant` tail).
* **Reuse is exact-match per document.** Two documents share cached KV only if
  their token sequences are identical, so keep the separators and whitespace of a
  given document stable across requests.

A worked example against the Block-Attention Llama-3 RAG template is in
[`lazy_block_infer.py`](../lazy_block_infer.py); the canonical serving-side
construction is `LazyRAG` in [`benchmarks/rag/rag.py`](../benchmarks/rag/rag.py).

## 4. Online / async serving

`AsyncLLM` is patched the same way and takes **`document_seq`** (singular — one
request at a time):

```python
import lazy.__vllm__  # noqa: F401
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

llm = AsyncLLM.from_engine_args(
    AsyncEngineArgs(model="hxia7/Llama-3.2-1B-Block-FT"))

async for out in llm.generate(
    prompt="Question: Which city is the capital of France? Answer:",
    sampling_params=SamplingParams(temperature=0, max_tokens=64),
    request_id="req-0",
    document_seq=docs,
):
    ...
```

A minimal FastAPI wrapper around this is
[`benchmarks/serve_demo.py`](../benchmarks/serve_demo.py); one model per process,
one process per method.

> Note: the OpenAI-compatible server is not extended — the OpenAI schema has no
> place to put `document_seq`. Serve through the async engine, or through a thin
> wrapper like `serve_demo.py`.

## 5. Configuration

### Engine arguments

Ordinary vLLM arguments, with two constraints:

* **`enable_prefix_caching` must stay on** (vLLM V1's default). Per-document reuse
  *is* prefix caching, applied per document; with it off, documents never register
  as ready.
* **`n` must be 1 for requests with documents** (document-free requests are
  unaffected). Parallel sampling is not implemented on the lazy path.
* **Documents cannot be combined with LoRA or pooling params** — both are
  rejected with an error explaining why (see [design.md §9](./design.md#9-scope-and-limitations)).
* `block_size` (default 16) sets the padding granularity — every document is
  rounded up to a whole number of blocks.

### The variant switch

```bash
LAZY_ATTENTION_VARIANT=lazy    # default: deferred key rotation
LAZY_ATTENTION_VARIANT=mepic   # position-independent cache baseline
```

One variable selects the kernel set, the RoPE behaviour and the scheduler's
rotation metadata together. Aliases: `lazy_attn`, `lazy_attention` /
`nope`, `position_independent`. See §6 of [design.md](./design.md) for the
difference, and [`lazy/utils/variants.py`](../lazy_attn/lazy/utils/variants.py)
for the authoritative list of every switch below.

### Tuning and profiling

Booleans accept `1/true/yes/on`.

| variable | effect |
|---|---|
| `NO_LAZY` | clear the per-request lazy flag in the attention wrapper. **A negative control, not a baseline** — see below |
| `LAZY_FORCE_SPLIT_DECODE` | use the lazy-only decode kernel when the whole batch is lazy |
| `LAZY_DECODE_IGNORE_Q_MASK` | drop the document padding mask (measurement only — changes results) |
| `LAZY_DECODE_COMPUTE_COS_SIN` | compute cos/sin in-kernel instead of loading `cos_sin_cache` |
| `MEPIC_FIRST_BLOCK_RECOMPUTE` | recompute each document's first block instead of reusing it (`mepic` only) |
| `MEPIC_FORCE_FP32_ROTARY` | do the in-kernel rotation in fp32 |

The decode switches are read **per call**, so a benchmark can flip them between
runs; each becomes a Triton constexpr, so a new value compiles a new kernel.

> **`NO_LAZY` does not produce an ordinary-vLLM baseline.** It only clears the
> flag the kernel branches on. By then the scheduler has already merged the
> request and its documents are already cached document-locally, so the vanilla
> branch applies neither the per-document query rotation nor `q_mask` — the
> attention is wrong, not vanilla. It is useful for isolating the kernel's cost,
> and for confirming a regression comes from the rotation path. For a real
> baseline, send the same documents inline in the prompt with no `document_seqs`
> (which is what `scripts/validate_lazy.py` and the `baseline` benchmark SUT do).

| variable | logs |
|---|---|
| `LAZY_SHARED_KV_PROFILE` (+ `..._MIN_REQS`, default 32) | shared-KV scheduling stats |
| `LAZY_PACKED_BLOCK_PROFILE` | packed block-table rebuild timings |
| `LAZY_PROFILE_ATTN_BACKEND` | per-layer attention timings |
| `LAZY_DECODE_WRAPPER_PROFILE` | decode wrapper timings |
| `DEBUG_LAZY=1` | dump padded document token ids and lengths at preprocessing |

Document-KV hit ratio is logged by the cache manager every 200 queries as
`[LAZY_DOC_KV_HIT] hits=… queries=… ratio=…`.

## 6. Tests

```bash
pytest lazy_attn/tests -c lazy_attn/tests/pytest.ini
```

Tests default to the 1B `hxia7/Llama-3.2-1B-Block-FT` so they fit on a single
consumer GPU. Set `LAZY_TEST_MODEL=ldsjmdy/Tulu3-Block-FT` to reproduce the
paper's 8B accuracy numbers.

Three kernel/utility modules import helpers from vLLM's own `tests.*` package,
which the wheel does not ship; they report as skipped, with the reason, unless you
are running against a vLLM source checkout.

The end-to-end smoke test is `scripts/validate_lazy.py`: it answers the same
questions three ways — documents inlined, documents through `document_seqs`, and
the same documents **reordered** — and requires every answer to contain the fact
its documents support. A broken rotation or a mis-addressed block produces fluent
text that no longer answers the question, which is exactly what it catches.

## 7. Benchmarks and the demo

```bash
# Live side-by-side A/B in tmux (lazy on :8001, prefix caching on :8002)
bash scripts/demo/lazy_vs_baseline_demo.sh

# Record the comparison GIF
bash scripts/demo/record_race_gif.sh
```

The main benchmark entrypoint is `benchmarks/benchmark_rag_serving.py`, driven by
the thin shell layer in `scripts/benchmark/`; dataset and SUT aliases live in
`scripts/benchmark/bench_consts.sh`. See [benchmarks/INTRO.md](../benchmarks/INTRO.md).

Note that the benchmark harness selects the lazy SUT with
`VLLM_USE_LAZY_ATTENTION=1`, which is a **benchmark-side** switch: it tells
`benchmarks/rag/rag.py` to `import lazy.__vllm__` at import time. It is not read
by the `lazy` package itself.

## 8. Troubleshooting

**Answers are fluent but wrong / ignore the documents.** Most often the model:
LazyAttention encodes each document independently, so it needs a block-fine-tuned
checkpoint (`hxia7/Llama-3.2-1B-Block-FT`, `ldsjmdy/Tulu3-Block-FT`). Otherwise
check that the documents are not *also* inlined in `prompts`.

**A request hangs in the waiting queue.** Documents never became ready — check
that prefix caching is enabled, and that KV cache space is large enough to hold
every document of a request at once.

**`Unsupported lazy attention variant '…'`** — `LAZY_ATTENTION_VARIANT` accepts
only `lazy`/`mepic` and their aliases.

**`n > 1 is not supported` with documents** — parallel sampling is not
implemented on the lazy path; issue *n* separate requests. Document-free
requests are unaffected.

**Documents + LoRA is refused.** The document prefill runs the base model, and
both sides of the document hash omit the adapter — so the request would not
fail, it would quietly attend to base-model KV under a LoRA query and answer
wrongly. Send the documents inline in the prompt, or drop the adapter.

**Triton compile errors on RTX 50-series (`sm_120`).** Triton 3.3 cannot compile
`tl.dot` for Blackwell; `bash scripts/install.sh --check` reports this, and the
installer upgrades Triton to ≥ 3.4.

**Reuse is lower than expected.** Reuse is exact-match on a document's token ids.
Trailing whitespace, a separator added on one path but not another, or a document
rebuilt from a different template all produce different blocks. `DEBUG_LAZY=1`
prints the padded token ids; `[LAZY_DOC_KV_HIT]` gives the running hit ratio.
