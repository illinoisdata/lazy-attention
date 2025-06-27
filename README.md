# Lazy-Attention

It is the full codebase for lazy-attention project. This version is for GH200.

Include:

- LazyAttn ([lazyattn](./lazyattn/))
- CacheBlend ([cacheblend](./cacheblend/))
- PromptCache ([promptcache](./promptcache/))
- Original vLLM ([vllm_proj](./vllm_proj/))

Details in each folder.

All experiments in [benchmarks](./benchmarks/).


**Note**: LazyAttn depends new version of vLLM while CacheBlend uses older version. Two independent envs are needed.