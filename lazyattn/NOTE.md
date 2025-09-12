# Scheduler

In `block_pool.py`, `get_new_blocks` and `touch` will increase the `ref_cnt` of a list of KV blocks.
In `kv_cache_manager.py`, the reference counter is updated in `allocate_slots`.

Allocate slots after we merge requests