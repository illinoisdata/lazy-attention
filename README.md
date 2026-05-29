
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