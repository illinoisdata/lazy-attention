# Lazy Attention Intro

The lazy-attention integration has two distinct layers:

1. Kernel-adjacent execution changes inside `lazy_attn/lazy/attention/`, `lazy_attn/lazy/model_executor/`, and scheduler/worker overrides.
2. A monkey-patch layer that swaps selected vLLM classes and methods with the lazy versions.

After cleanup, the patch orchestration is centralized in `lazy_attn/lazy/vllm_patch.py`.

- `apply_attention_patches()` patches only the attention-path pieces needed in-process.
- `apply_all_patches()` patches the full frontend + engine + worker + scheduler stack.
- `__vllm__.py` imports that module and applies the full patch set on import.
- `ctxmgr.py` reuses the same patch registry for explicit context-managed patching.

The reason to keep the patch layer separate from Triton code is performance safety:

- Triton kernels and the hot-path attention implementation stay in their original files.
- The patch registry only swaps references to those implementations.
- This makes it easier to reason about correctness and revert behavior without touching the kernel code.

When tracing a lazy request through the system, the high-level path is:

1. `lazy.__vllm__` patches vLLM.
2. `LazyLLM` / `LazyLLMEngine` / `LazyProcessor` convert document sequences into the extended request format.
3. `LazyEngineCoreProc`, `LazyScheduler`, and `LazyGPUModelRunner` carry the extra request metadata through scheduling and execution.
4. The patched attention stack eventually reaches the custom Triton-backed forward path.

If you need to modify behavior later, change the concrete lazy implementation first, and only touch `vllm_patch.py` when the set of patched symbols changes.
