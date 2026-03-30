from __future__ import annotations  # isort:skip

from block_attn_vllm.scheduler import BlockAttnScheduler
from lazy.vllm_patch import apply_all_patches

apply_all_patches(scheduler_cls=BlockAttnScheduler)
