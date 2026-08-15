
<div style="text-align: center;">
  <img src="/docs/assets/logo.svg" alt="la-logo"/>
</div>


Codebase for the LazyAttention project. It bundles the three components that
share the **same vLLM backend** (the modern vLLM in [vllm_proj](./vllm_proj/)),
so they can be compared apples-to-apples on one engine:

- **LazyAttention** ([lazy_attn](./lazy_attn/)) — defers positional encoding and
  caches one position-agnostic KV copy per document, reused regardless of slot.
- **BlockAttention** ([block_attn_vllm](./block_attn_vllm/)) — block-diagonal
  attention over independently-encoded document blocks, integrated into vLLM.
- **Original vLLM** ([vllm_proj](./vllm_proj/)) — the unmodified backend both
  build on, and the source of the stock baselines (prefix caching / full recompute).

Details in each folder. All experiments in [benchmarks](./benchmarks/).

## Install

```bash
bash scripts/install.sh --venv .venv   # ~2 minutes
source .venv/bin/activate
python scripts/validate_lazy.py        # end-to-end check
```

That is the whole install. LazyAttention and BlockAttention add no C++/CUDA of
their own — they are Python monkey patches over vLLM plus Triton kernels that
are JIT-compiled at runtime — so the prebuilt vLLM wheel already contains every
compiled kernel they need. The installer pins the combination that works:

| component    | version        | why |
|--------------|----------------|-----|
| vLLM         | 0.9.2          | oldest release whose wheels ship `sm_120` kernels while still exposing the internals we patch |
| torch        | 2.7.0+**cu128**| the default PyPI torch 2.7.0 is a cu126 build with no `sm_120` kernels |
| transformers | 4.53.2         | vLLM 0.9.2 predates the transformers 5.x config registry |
| Triton       | ≥ 3.4 on `sm_120` | Triton 3.3 cannot compile `tl.dot` for Blackwell |

Useful flags: `--bench` also installs the benchmark dependencies, `--check`
reports on the environment without installing, and `--source` builds vLLM from
source (see [vllm_proj/install.sh](./vllm_proj/install.sh) — only needed if you
are modifying vLLM's own C++/CUDA).

### Hardware

Anything from Ampere through Blackwell: `sm_80` (A100), `sm_89` (L40S),
`sm_90` (H100/GH200) and `sm_120` (RTX 50-series). On consumer Blackwell the
installer additionally upgrades Triton, which `scripts/install.sh --check`
will tell you about up front.

### Tests

```bash
pytest lazy_attn/tests -c lazy_attn/tests/pytest.ini
```

Tests default to the 1B `hxia7/Llama-3.2-1B-Block-FT` so they fit on a single
consumer GPU. Set `LAZY_TEST_MODEL=ldsjmdy/Tulu3-Block-FT` to reproduce the
paper's 8B accuracy numbers.

Three kernel/utility modules import helpers from vLLM's own `tests.*` package,
which the wheel does not ship. They report as skipped, with the reason, unless
you are running against a vLLM source checkout.

### Variants

`LAZY_ATTENTION_VARIANT=lazy|mepic` selects the attention implementation
(default `lazy`). That one variable picks the kernel set, the RoPE behaviour
and the scheduler's rotation metadata together;
[lazy/utils/variants.py](./lazy_attn/lazy/utils/variants.py) documents it and
every tuning/profiling switch in one place.

## Demo: Lazy-Attn vs Prefix Caching

<div style="text-align: center;">
  <img src="/docs/assets/lazy_vs_prefix_demo.gif" alt="lazy-vs-prefix-caching demo"/>
</div>

Both serve the **same** retrieved documents, but in a **new order** per request.
Prefix caching can only reuse a contiguous prefix, so a reordering forces it to
recompute the rest; Lazy-Attn caches one position-agnostic copy per document and
reuses every block regardless of slot — reaching the first token **3.3× sooner**
(201 ms vs 655 ms here, 8B `Tulu3-Block-FT`, 2WikiMultihopQA), with an identical
answer. The gap grows with context length.

Run it yourself (a tiny FastAPI server wraps each `RAG` SUT; one model per process):

```bash
# Live side-by-side A/B in tmux (lazy on :8001, prefix caching on :8002, client pane)
bash scripts/demo/lazy_vs_baseline_demo.sh

# Record the GIF above. Locally (1B, directional timing):
bash scripts/demo/record_race_gif.sh
# On the 8B (coherent answers) via slurm:
sbatch scripts/demo/record_race_gif.slurm        # --export=ALL,DEMO_RECORD_INDEX=N for other questions
```

See [benchmarks/serve_demo.py](./benchmarks/serve_demo.py) (server),
[benchmarks/demo_race.py](./benchmarks/demo_race.py) (capture + GIF render), and
[scripts/demo/](./scripts/demo/) (launchers).


## Citation

If this repo is helpful for you research, please cite our paper.

```
@inproceedings{
2026lazyattention,
title={LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding},
author={Haocheng Xia and Mihir Pamnani and Hanxi Fang and Supawit Chockchowwat and Yongjoo Park},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=M9kHwqreN9}
}
```

## Contact us

- For technical questions and feature requests, please use GitHub Issues
- For collaborations, please contact the authors.