# LazyAttention — Design

How the `lazy` package turns vLLM into an engine that caches **one
position-agnostic KV copy per document** and reuses it in any slot of any later
prompt.

This document covers the *what* and *why*. For how to run it, see
[usage.md](./usage.md).

---

## 1. The problem

A RAG prompt is a list of retrieved documents followed by a query:

```
[doc A][doc B][doc C][query]
```

vLLM's prefix caching reuses a **contiguous prefix from position 0**. The next
request retrieves the same three documents in a different order —
`[doc C][doc A][doc B][query]` — and the cache hit ends at the first token that
differs. Everything after it is recomputed, even though every document in it was
computed minutes ago.

The reason is RoPE. When vLLM writes a key to the KV cache, that key has already
been rotated by its absolute position in the sequence. A cached key is therefore
only valid at the exact offset where it was produced: doc B's KV computed at
positions `[512, 768)` is meaningless at positions `[0, 256)`.

## 2. The idea

Cache each document **as if it started at position 0**, independent of the prompt
it appears in, and fix up the positions at attention time by rotating the
**query** instead of the keys.

This works because a RoPE attention score depends only on the *difference* of the
two positions:

```
    <R(p_q) q, R(p_k) k>  =  <R(p_q - p_k) q, k>
```

So rotating one query token backwards by Δ is algebraically identical to rotating
every key of that document forwards by Δ — but it costs **one rotation per query
token per document**, not one per cached key. The documents' KV never has to be
touched, which is what makes it reusable in any slot.

The price is that documents no longer see each other during encoding (each is
encoded alone, from position 0). That is a *model* change, not just an engine
change, so LazyAttention runs on **block-fine-tuned checkpoints** (`*-Block-FT`)
which were trained with exactly this block-diagonal document encoding.

## 3. Request lifecycle

A lazy request carries its documents beside the prompt rather than inside it:

```python
llm.generate(prompts=[query], document_seqs=[[doc_a, doc_b, doc_c]])
```

```
LazyLLM.generate(document_seqs=...)                    entrypoints/llm.py
  └─ LazyLLMEngine.add_request(document_seq=...)       engine/llm_engine.py
       └─ LazyProcessor.process_inputs                 engine/processor.py
            tokenize each doc, drop BOS, right-pad to a
            multiple of block_size, hash the doc sequence
       └─ EngineCoreRequest(+documents_token_ids_padded,
                            document_lens, document_lens_padded,
                            document_seq_hash)         engine/__init__.py
            └─ LazyRequest                             request.py
                 └─ LazyScheduler.schedule()           core/sched/scheduler.py
                      ├─ documents missing? spawn document requests
                      ├─ documents ready?   merge + build q_offset/q_mask
                      └─ NewRequestData(is_lazy, q_offset, q_mask)
                           └─ LazyGPUModelRunner       worker/gpu_model_runner.py
                                pack metadata into per-request buffers
                                └─ patched attention → Triton kernels
```

### 3.1 Preprocessing (`engine/processor.py`)

For each document in `document_seq`:

* tokenize it on its own, and **drop the leading BOS** — a document is a fragment,
  not a sequence start;
* **right-pad to a multiple of `block_size`** with the tokenizer's pad token, so
  each document owns a whole number of KV blocks and never shares a block with its
  neighbour (a shared block would be position-dependent again);
* record `document_lens` (true) and `document_lens_padded` (allocated).

The query prompt is tokenized normally (also minus BOS when documents are
present) and stays in `prompt_token_ids`. Finally the whole padded document
sequence is hashed into `document_seq_hash`, which later seeds the query's block
hashes (§3.4).

### 3.2 Scheduling: documents as first-class requests

`LazyScheduler` sees a waiting request with documents and asks
`is_doc_ready()` — one prefix-cache lookup per document
(`LazyKVCacheManager.get_computed_blocks_docs`). Any document that is not fully
cached gets a **synthetic document request** spawned for it:

```python
LazyRequest.document_request(doc_idx)   # request_id = f"{parent}_d{idx}"
```

pushed to the **head** of the waiting queue, because it is what blocks the parent.
A document request:

* prefills only — `max_tokens = 1`, and it reserves **no lookahead slots**
  (`LazyKVCacheManager.allocate_slots`);
* may be served **entirely** from cache. The base manager caps a hit at
  `num_tokens - 1` so the last token can produce logits; a document never samples,
  so the cap is lifted (`get_computed_blocks`);
* deliberately does **not** forward `lora_request` — and that is exactly why
  documents + LoRA is **rejected at input validation** rather than left to run.
  Both the lookup and the write go through this method, so their hashes agree
  and the parent *would* proceed — attending to document KV computed by the base
  model while its own query and decoding run under the adapter. Nothing errors;
  the answer is just wrong, which is the worst way to fail.

Once every document is ready, `merge_documents()` splices the padded document
tokens in front of the query and the request becomes an ordinary request whose
prefix happens to be fully cached. The per-document hits are prepended, in
document order, to the request's computed-block list, and their block hashes are
prepended to `req_to_block_hashes` so subsequent caching lines up.

### 3.3 Why one lookup per document is the whole trick

`get_computed_blocks_docs` runs `find_longest_cache_hit` **per document**, not
once over the merged prompt. Reordering the documents permutes the lookups; it
does not truncate them. That is the entire difference from prefix caching, and it
is why the reuse rate is order-independent.

### 3.4 Hashing (`core/kv_cache_utils.py`)

Two rules, and both matter:

* **Documents** are hashed by handing the *same synthetic document request* to
  vLLM's own `hash_request_tokens`. The lookup side and the write side therefore
  cannot disagree — whatever upstream folds into a block hash today (cache salt)
  or adds tomorrow is applied once, to one object. Crucially the chain starts from
  `None`, so a document's hashes are independent of whatever precedes it in the
  prompt. That is what makes it reusable at any offset.
* **Query blocks** use `hash_request_tokens_with_doc_hash`, which seeds the chain
  with `document_seq_hash` instead of `None`. Identical queries behind different
  document sets must not alias, and they don't.

  The **boundaries** are part of that seed, not just the tokens. One 32-token
  document and two 16-token ones can flatten to the same token sequence, but
  they are encoded block-diagonally into different KV, so seeding them alike
  would let a query block computed against one be reused for the other. The
  hash is taken over the documents as a tuple of tuples for that reason.

## 4. The rotation metadata

This is the part that carries the position fix-up from the scheduler down to the
kernel. It is computed once per request, in
`scheduler.py::metadata_for_lazy_attention`, as two per-block arrays.

### 4.1 The offset, derived

Let block size be `B`, and document `d` have true length `L_d` and padded length
`P_d = ceil(L_d / B) * B`. Write `P = Σ P_d` (the merged document region) and
`P_pad = Σ (P_d − L_d)` (all padding).

* **Actual positions.** After `merge_documents()`, vLLM lays out
  `[padded doc 0]…[padded doc n−1][query]` and assigns positions `0 … P+T−1`, so
  query token `t` is rotated at `P + t`.
* **Stored key angles.** Document `d`'s keys came from a standalone document
  request, so its key `i` carries angle `i` — starting from 0, always.
* **Wanted layout.** The documents should appear back-to-back *with padding
  removed*: doc `d`'s token `i` at `S_d + i` where `S_d = Σ_{e<d} L_e`, and query
  token `t` at `Σ_e L_e + t`.

The key angle is `i` but should be `S_d + i`, so de-rotate the query by `Δ_d`:

```
    (P + t) − Δ_d  =  (Σ_e L_e) + t        ⟹    Δ_d = P_pad + Σ_{e<d} L_e
```

which is exactly the accumulator in the code — it starts at the total padding and
grows by each document's *true* length:

```python
abs_rot_pos = int(sum(padding_lens))                  # = P_pad
for doc_idx in range(num_docs):
    q_offset[blocks of doc d] = abs_rot_pos + 1       # = Δ_d + 1
    q_mask[last block of doc d] = padding_lens[d]
    abs_rot_pos += request.document_lens[doc_idx]     # += L_d
q_offset[query block] = 1                             # Δ = 0
```

### 4.2 Encoding

`q_offset` is stored per block with a **+1 bias**, which buys two sentinels:

| value  | meaning                                                   |
|--------|-----------------------------------------------------------|
| `0`    | untouched — non-lazy request, or "leave Q as it is"        |
| `1`    | reset to the original Q (the query's own block, Δ = 0)     |
| `v>1`  | de-rotate Q by `v − 1`                                     |

Sentinel `0` means "keep the rotation currently in effect", which is why the
arrays are only `num_doc_blocks + 1` long: the first query block resets Q with a
`1`, and every later query block inherits it as a zero. Rows of the per-request
buffer are zero-filled beyond that, so a query of any length is covered.

`q_mask` is non-zero only on a document's **last** block, where it holds that
document's pad-token count. The kernel narrows that block's key mask by exactly
that many slots, so padding never contributes to a score.

### 4.3 What the kernel does with it

In the decode kernel (`attention/ops/models/llama_v1.py`), each program walks the
blocks of one sequence and re-rotates Q **only when `q_offset` changes** — that
is once per document, not once per block. The rotation is always rebuilt from the
original `Q_full` rather than applied incrementally, so error cannot accumulate
across a long document list, and it is the inverse rotation
(`q1·cos + q2·sin`, `−q1·sin + q2·cos`).

cos/sin are **loaded** from the model's `cos_sin_cache` by default; this decode
kernel is occupancy-bound, and loading measured faster than computing. The
in-kernel compute path exists behind `LAZY_DECODE_COMPUTE_COS_SIN=1`, with the
model's RoPE parameters (base, Llama-3 scaling factors) forwarded as constexpr via
`rope_meta_from_layer` so the computed values equal the table's.

### 4.4 The packed block table

The decode kernel needs three numbers per block: physical block index, `q_offset`,
`q_mask`. Rather than three loads, `LazyGPUModelRunner` maintains a persistent
`int64` table — one **per KV cache group**, since a block id only means anything
inside the group that allocated it — with all three packed:

```
[ physical_block_idx : 32 | q_offset : 16 | q_mask : 16 ]
```

Each layer is handed the table for its own group (`layer_name → group`); the
rotation tensors beside it are group-independent and shared.

It is rebuilt in full when the batch composition changes (additions, removals, or
attention-backend reordering — detected by comparing the `req_ids` tuple, plus a
forced rebuild whenever a new request arrives, which covers an id being reused),
and row-wise when only block counts grew.

## 5. Execution path

The attention stack is patched end to end so the RoPE tables and the lazy metadata
reach the kernel:

| layer | file | what changes |
|---|---|---|
| `LlamaAttention.forward` | `model_executor/models/llama.py` | runs RoPE, then passes `inv_freq`, `cos_sin_cache`, `rotary_dim`, `is_neox_style` and a cached `rope_meta` down to the attention layer |
| `RotaryEmbedding` / `Llama3RotaryEmbedding` | `model_executor/layers/rotary_embedding.py` | materialises `inv_freq` as a buffer; under `mepic`, rotates **Q only** |
| `Attention.forward` | `attention/layer.py` | accepts and forwards those arguments; dispatches through the custom op `unified_lazy_attention_with_output` |
| `CompilationConfig.set_splitting_ops_for_v1` | `attention/layer.py` | registers that op as a splitting op so piecewise CUDA graphs still split at attention |
| `TritonAttentionImpl.forward` | `attention/backends/triton_attn.py` | writes K/V to the paged cache as usual, then calls the lazy kernels with `is_lazy`, `lazy_variant`, `q_offset`, `q_mask`, `packed_block_table` |

Inside `chunked_prefill_paged_decode`:

* `max_query_len > 1` → the lazy `context_attention_fwd` (prefill / chunked
  prefill), which applies the same per-block Q rotation while the query region is
  being computed against cached document blocks;
* the decode kernel `kernel_paged_attention_2d_llama` always runs; a lazy-only
  variant (`..._lazy_only`, no non-lazy branch) is selected when every request in
  the batch is lazy and `LAZY_FORCE_SPLIT_DECODE=1`.

Because `is_lazy` is per request, lazy and ordinary requests coexist in one batch:
non-lazy rows take the untouched path through the same kernel.

## 6. Variants

`LAZY_ATTENTION_VARIANT` picks the kernel set, the RoPE behaviour and the
scheduler's metadata **together** — one switch, documented in
[`lazy/utils/variants.py`](../lazy_attn/lazy/utils/variants.py):

| | `lazy` (default) | `mepic` (aliases `nope`, `position_independent`) |
|---|---|---|
| RoPE in the model | rotates Q **and** K normally | rotates **Q only** |
| What is cached | keys rotated at their document-local position | keys with no rotation at all |
| Who fixes positions | kernel de-rotates **Q** per document block | kernel rotates the cached **K** at its slot |
| Kernels | `lazy/attention/ops/` | `lazy/attention/old_ops/` |
| `q_offset` | `Δ_d + 1` per block | all zeros (keys rotate to absolute slot) |
| Cost | one Q rotation per document | one K rotation per cached key, every step |

`mepic` is the position-independent-cache baseline the paper compares against; it
is kept because it is the honest alternative implementation of the same idea, and
because it shares the scheduler and cache paths so the comparison isolates the
kernel.

## 7. The patch layer

LazyAttention adds **no C++/CUDA**. It is Python monkey patches over vLLM plus
Triton kernels JIT-compiled at runtime, which is why the prebuilt vLLM wheel is
enough to install it.

All patching is centralised in
[`lazy/vllm_patch.py`](../lazy_attn/lazy/vllm_patch.py) as a registry of
`PatchTarget(module, attribute, replacement)` in three groups:

* `attention_targets()` — the RoPE classes, `Attention.forward`,
  `TritonAttentionImpl.forward`, `LlamaAttention.forward`, splitting ops;
* `frontend_targets()` — `LLM`, `Processor`, `LLMEngine`, `AsyncLLM`,
  `EngineCoreRequest`, `EngineCoreProc`, `Request`, `CachedRequestState`,
  `GPUModelRunner`;
* `scheduler_targets()` — `Scheduler`.

Two entry points use it:

* `apply_all_patches()` — the whole stack. `import lazy.__vllm__` calls it, which
  is the one-line way to enable LazyAttention.
* `apply_attention_patches()` — the attention path only, which is what the engine
  core subprocess needs. `LazyEngineCoreProc.run_engine_core` re-applies it inside
  the spawned process, since patches applied in the parent do not survive the fork
  boundary under multiprocessing.

`ctxmgr.LazyAttentionContextManager` reuses the same registry to scope the
**attention patches only** (`with LazyAttentionContextManager(): ...`), with
matching revert paths. It is not a scoped `apply_all_patches`: inside the block
the frontend is still stock vLLM, so it serves kernel work, not document
requests.

Every patch also records the original, so `revert_all_patches()` restores stock
vLLM in place. Applying the patch set additionally forces
`VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1`, since the lazy kernels are Triton.

The split between the kernels and the patch registry is deliberate: the hot-path
implementations stay in their own files and the registry only swaps references to
them, so correctness can be reasoned about — and reverted — without touching
kernel code.

## 8. Design decisions worth knowing

**Delegate, don't fork.** `LazyScheduler`, `LazyKVCacheManager` and
`LazyGPUModelRunner` call `super()` and post-process. vLLM rewrites
`_prepare_inputs` and `_update_states` most releases; forking them means
re-porting every bump. `NewRequestData` subclasses upstream's dataclass for the
same reason — fields added upstream are inherited, not re-copied.

**Defaults on the extra fields.** `EngineCoreRequest`'s document fields all
default to `None` so vLLM's own call sites, which know nothing about documents,
keep constructing it fine once it is patched in.

**Keyword arguments across the vLLM boundary.** `process_inputs` is always called
by keyword: upstream has inserted new *positional* parameters between releases
(`tokenization_kwargs` in 0.9.x), which silently shifts positional arguments.

**Padding is a real cost.** Each document is rounded up to a whole number of
blocks, so a 20-token document with `block_size=16` occupies 32 slots. That is the
price of making a document independently addressable; the padded slots are masked
out of every score by `q_mask`.

## 9. Scope and limitations

* **Models**: Llama-family only — `LlamaAttention.forward` is what gets patched.
  Answer quality assumes a block-fine-tuned checkpoint; a stock checkpoint runs
  but was never trained on block-diagonal document encoding.
* **Backend**: Triton (`TRITON_ATTN_VLLM_V1`), forced on patch. Cascade attention
  is asserted off.
* **Prefix caching must stay enabled** — per-document reuse *is* prefix caching,
  applied per document.
* **Unsupported, and refused rather than run**: `n > 1` **with documents**
  (document-free requests keep upstream's parallel sampling), documents + LoRA,
  documents + pooling. Encoder-decoder models, prompt adapters and tracing are
  unsupported upstream on this path.
* **KV cache groups** must all use the scheduler's block size — the rotation
  metadata is per block at one block size. Each group gets its own packed block
  table; a group on a different block size is refused.
* **Prompt shape**: documents are always merged **in front of** the prompt, in the
  given order. A preamble that must precede the documents has to be the first
  element of `document_seqs`.

## 10. Map of the code

```
lazy/
  __vllm__.py             import this to patch vLLM              (§7)
  vllm_patch.py           the patch registry                     (§7)
  ctxmgr.py               scoped patching, attention path only
  request.py              LazyRequest, document_request()        (§3.2)
  entrypoints/llm.py      LazyLLM.generate(document_seqs=...)    (§3)
  engine/
    __init__.py           EngineCoreRequest + document fields    (§3.1)
    processor.py          tokenize / pad / hash documents        (§3.1)
    llm_engine.py         sync add_request(document_seq=...)
    async_llm.py          async add_request/generate(document_seq=...)
    core.py               LazyEngineCoreProc (re-patch in subproc)
  core/
    sched/scheduler.py    doc readiness, merge, q_offset/q_mask  (§3.2, §4)
    sched/output.py       NewRequestData + lazy fields
    kv_cache_manager.py   per-document cache lookups             (§3.3)
    kv_cache_utils.py     document / query block hashing         (§3.4)
  worker/
    gpu_model_runner.py   metadata buffers, packed block table   (§4.4)
    gpu_input_batch.py    CachedRequestState + lazy fields
  model_executor/
    models/llama.py       pass RoPE tables to attention          (§5)
    layers/rotary_embedding.py  inv_freq buffer, Q-only mode     (§6)
    rope/cos_sin.py       in-kernel RoPE cos/sin                 (§4.3)
  attention/
    layer.py              patched Attention.forward + custom op  (§5)
    backends/triton_attn.py   kernel dispatch                    (§5)
    ops/                  lazy variant kernels                   (§6)
    old_ops/              mepic variant kernels                  (§6)
  utils/variants.py       every runtime switch, documented       (§6)
```
