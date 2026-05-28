
<div style="text-align: center;">
  <img src="/docs/assets/logo.svg" alt="la-logo"/>
</div>


It is the full codebase for lazy-attention project. This version is a local dev branch tested on a desktop equiped with one 5070ti.

> Full attention base model: meta-llama/Llama-3.2-1B or meta-llama/Llama-3.2-1B-Instruct

> Block attention model: hxia7/Llama-3.2-1B-block-FT

Include:

- LazyAttention ([lazy_attn](./lazy_attn/))
- BlockAttention ([block_attn_vllm](./block_attn_vllm/))
- CacheBlend ([cacheblend](./cacheblend/))
- PromptCache ([promptcache](./promptcache/))
- Original vLLM ([vllm_proj](./vllm_proj/))

Details in each folder.

All experiments in [benchmarks](./benchmarks/).

## Environments

The baselines need **independent** envs (do not reuse the root `.venv`, which
holds the new vLLM that LazyAttention/BlockAttention build on):

- **LazyAttention / BlockAttention** — root `./.venv` (torch 2.7.0+cu128, new vLLM). Runs on this desktop (sm_120 / RTX 5070 Ti).
- **PromptCache** — `promptcache/.venv`:
  ```bash
  cd promptcache && uv venv --python 3.10 .venv
  VIRTUAL_ENV=.venv uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  VIRTUAL_ENV=.venv uv pip install -e .          # deps pinned in requirements.txt
  .venv/bin/python smoke_test.py                 # end-to-end check (passes on this desktop)
  ```
  Uses transformers 4.36.2 (it imports `transformers.file_utils`, removed ~4.40),
  so it targets Llama-2-architecture models; `smoke_test.py` uses TinyLlama.
- **CacheBlend** — `cacheblend/.venv` (torch 2.2.1 + cu121, bundled vLLM 0.4.1 in `vllm_blend/`, needs a `vllm._C` source build).
  **Not runnable on this desktop**: torch 2.2.1 supports only sm_50–sm_90, but the
  RTX 5070 Ti is sm_120 (Blackwell) — GPU ops fail with "no kernel image is
  available for execution on the device". Run it on an sm≤9.0 GPU (A100/H100/etc.),
  or port CacheBlend to a modern vLLM.

<!-- ## Benchmark Notes

- `exp1` convenience scripts:
  - `scripts/benchmark/exp1_lazy_bench.slurm`: runs `lazyrag` on `2wikimqa`
  - `scripts/benchmark/exp1_mepic_bench.slurm`: runs `lazyrag` with `LAZY_ATTENTION_VARIANT=mepic`
  - `scripts/benchmark/exp1_baseline_bench.slurm`: runs `baseline` on `2wikimqa`

- `mepic` benchmark default:
  - `scripts/benchmark/exp1_mepic_bench.slurm`
  - `scripts/benchmark/exp4_mepic_bench.slurm`
  - both default to `MEPIC_FORCE_FP32_ROTARY=1`
  - this is the conservative MEPIC baseline used in our comparisons

- Shared-KV fairness setting:
  - `scripts/benchmark/shared_kv_scale_baseline.slurm` sets `BASELINE_PREPARE_PREFIX_CACHE=1`
  - this makes baseline warm prefix cache during `add_doc_async()`
  - the default for `BaselineLazyRAG` remains off, so regular benchmark scripts are unaffected

- Debug timing:
  - `VLLM_LOG_MODEL_FORWARD_TIME=1` enables per-step model forward timing logs
  - this path uses explicit CUDA synchronization and can perturb benchmark numbers
  - keep it unset (default) for performance runs -->


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