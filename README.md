
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

<!-- 
**Note**: LazyAttn depends new version of vLLM while CacheBlend uses older version. Two independent envs are needed. -->

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