from __future__ import annotations  # isort:skip

import vllm.v1.worker.gpu_model_runner as _gmr
from vllm.v1.core.block_pool import BlockPool

from lazy.vllm_patch import apply_all_patches
from lazy.core.block_pool import cache_full_blocks as _lazy_cache_full_blocks

from block_attn_vllm.gpu_model_runner import BlockAttnGPUModelRunner
from block_attn_vllm.scheduler import BlockAttnScheduler

# Reuse the lazy frontend + metadata (q_offset = abs_rot_pos+1, q_mask). The
# block-attention scheduler adds copy-on-write of document blocks; the runner
# rotates those copies before the kernel instead of rotating the query inside
# it (so the canonical cached blocks stay pristine).
apply_all_patches(scheduler_cls=BlockAttnScheduler)

# Block-Attention redirects each query's document blocks to fresh copy-on-write
# blocks that are NOT registered in the prefix cache (they are request-private
# rotated scratch, never shared). Stock BlockPool.cache_full_blocks would walk
# back into that doc region and assert the copies have a block hash. The lazy
# variant skips the document region when hashing later (query/decode) blocks --
# it chains from request.document_seq_hash instead -- so bind it here. (It is
# stock cache_full_blocks plus that document override; see lazy/core/block_pool.)
BlockPool.cache_full_blocks = _lazy_cache_full_blocks

# Swap the lazy runner for the block-attention runner (must come after
# apply_all_patches, which installs LazyGPUModelRunner).
_gmr.GPUModelRunner = BlockAttnGPUModelRunner
