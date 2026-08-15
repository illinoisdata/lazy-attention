# vllm-proj — the unmodified backend

The stock vLLM that LazyAttention and BlockAttention both patch at runtime. It
is here for two reasons:

1. **the baseline.** Stock prefix caching and full recompute are what the paper
   compares against, and they have to run on the same engine as the lazy path
   for the comparison to mean anything.
2. **a source build**, for the rare case where you are changing vLLM's own
   C++/CUDA.

vLLM itself is **not vendored** here. `install.sh` clones it
(`https://github.com/vllm-project/vllm`) at `VLLM_COMMIT`, default **v0.9.2** —
the version the rest of the repo is pinned to, and the oldest release whose
wheels ship `sm_120` kernels while still exposing the internals the patches
reach into. A `vllm/` directory in this folder is that clone, and is not tracked.

## You probably do not need this

```bash
bash scripts/install.sh --venv .venv    # ~2 minutes, prebuilt wheel
```

is the documented path and the one that is tested. LazyAttention and
BlockAttention add no C++/CUDA of their own — they are Python monkey patches
plus Triton kernels JIT-compiled at runtime — so the prebuilt wheel already
contains every compiled kernel they need. A source build takes 20–60 minutes and
buys nothing unless you are editing vLLM's own extensions.

## Building from source

```bash
bash scripts/install.sh --venv .venv --source   # seeds torch, then calls install.sh here
# or, into an environment that already has a matching torch:
bash vllm_proj/install.sh
```

`install.sh` needs `torch` already installed (it builds against it), plus
`cmake`, `ninja` and a CUDA toolkit; it fails early and by name if any is
missing. Overrides:

| variable | default | |
|---|---|---|
| `VLLM_COMMIT` | `v0.9.2` | vLLM commit or tag to build |
| `CUDA_HOME` | auto-detected from `nvcc`, else `/usr/local/cuda` | CUDA toolkit root |
| `MAX_JOBS` | `nproc`, capped at 8 | parallel compile jobs |
| `NVCC_THREADS` | 4 | threads per `nvcc` |

It configures with CMake + Ninja, builds `_C` and `_vllm_fa2_C` (fa3 is skipped
deliberately — it needs far more memory to compile), installs those extensions,
then does an editable install with the compiled artifacts left in place. `ccache`
is used if present. The last step imports `vllm._C` so a build that silently
produced no extensions fails here rather than at the first kernel launch.

## What we changed

One thing: **`setup.py` gains a `NO_C=1` escape hatch** that empties
`ext_modules`. `install.sh` copies this file over the clone's own and builds the
extensions by hand first, so the editable install at the end must not build them
a second time. Nothing else in vLLM is modified — everything LazyAttention needs
is a runtime patch (see [../docs/design.md §7](../docs/design.md#7-the-patch-layer)).

## The rest of this folder

| file | |
|---|---|
| `validate.sh` | smoke-test the unmodified backend. For the lazy path use `scripts/validate_lazy.py` |
| `bench.sh` | serving benchmark against ShareGPT, with recorded A40 numbers in the comments |
| `tests/generate.py` | vLLM's offline `generate` example, pinned to the Triton backend |

> Note: `triton/` is an empty gitlink left over from an earlier layout, with no
> `.gitmodules` entry — `git submodule` commands will complain about it. Nothing
> reads it.

## See also

- [../docs/design.md](../docs/design.md) — how LazyAttention patches this backend
- [../docs/usage.md](../docs/usage.md) — installing and running the patched engine
- [../README.md](../README.md) — the three components and the version pins
