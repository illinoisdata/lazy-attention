# Benchmark Intro

`benchmarks/benchmark_rag_serving.py` is the main benchmark entrypoint. It does three jobs:

1. Load one benchmark dataset and normalize it into `RAGRequest` objects.
2. Build a concrete RAG system through `benchmarks/rag/rag.py::make_rag`.
3. Drive async request execution, then aggregate latency and throughput metrics.

The shell layer under `scripts/benchmark/` is intentionally thin after cleanup:

- `bench_consts.sh` is the catalog of dataset aliases and SUT aliases.
- `bench_lib.sh` is the shared runner logic used by `bench_exp1.sh`, `bench_exp4.sh`, `bench_exp5.sh`, and `scripts/validate.sh`.
- `bench_exp*.sh` only define experiment-specific knobs such as concurrency or ablation parameters.

The common execution flow is:

1. Choose a SUT alias like `lazyrag` or `baseline`.
2. Choose a dataset alias like `2wikimqa` or `blk_nq`.
3. Expand both aliases into CLI arguments with `make_sut_args` and `make_data_args`.
4. Apply environment toggles in `prepare_sut`.
5. Run `python benchmarks/benchmark_rag_serving.py ...`.

Dataset aliases support suffix overrides like `rand[n=32]` or `rand[rdl=1024]`. The parser for that mini-language still lives in `bench_consts.sh`, because the shell scripts consume it directly before dispatch.

If you need to add a new benchmark:

1. Add the dataset or SUT alias in `scripts/benchmark/bench_consts.sh`.
2. If it is a new experiment profile, add one thin wrapper next to `bench_exp1.sh`.
3. Keep experiment-specific parameters in the wrapper, not in the shared library.

If you need to change benchmark behavior itself, start from `benchmarks/benchmark_rag_serving.py`, then check `benchmarks/rag/rag.py` for the concrete RAG implementation path.
