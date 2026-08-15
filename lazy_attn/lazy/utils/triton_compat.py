"""Compatibility shim for running LazyAttention's Triton kernels on Blackwell.

LazyAttention forces the Triton attention backend, and Triton only learned to
emit `tl.dot` for consumer Blackwell (sm_120) in 3.4 -- with 3.3 the compiler
aborts in `TritonGPUAccelerateMatmul` with "computeCapability not supported".

torch 2.7 pins Triton 3.3, so an sm_120 run has to install Triton >= 3.4. That
combination is otherwise fine, except Triton 3.4 moved the launch hooks off
`CompiledKernel` (they now live in `triton.knobs.runtime`), while torch 2.7's
inductor still reads them as class attributes and dies with
`AttributeError: type object 'CompiledKernel' has no attribute
'launch_enter_hook'` the moment torch.compile is used.

Defining the attributes as None restores the "no hook installed" path inductor
already handles, which is exactly the behaviour we want. This is a no-op on
matched pairs (torch 2.8 + Triton 3.4, i.e. vLLM 0.10+), where the attributes
either exist or inductor no longer looks for them.
"""
from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_HOOKS = ("launch_enter_hook", "launch_exit_hook")


def apply_triton_launch_hook_shim() -> None:
    """Give `CompiledKernel` the launch-hook attributes torch's inductor reads."""
    try:
        from triton.compiler.compiler import CompiledKernel
    except ImportError:  # pragma: no cover - triton always present with vLLM
        return

    patched = [name for name in _HOOKS if not hasattr(CompiledKernel, name)]
    for name in patched:
        setattr(CompiledKernel, name, None)

    if patched:
        logger.debug(
            "Installed Triton launch-hook shim for torch inductor (%s)",
            ", ".join(patched),
        )
