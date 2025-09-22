# Scheduler

In `block_pool.py`, `get_new_blocks` and `touch` will increase the `ref_cnt` of a list of KV blocks.
In `kv_cache_manager.py`, the reference counter is updated in `allocate_slots`.

Allocate slots before we merge requests, it will simplify the logic of `cache_full_blocks`



## Appendix

Example of `q_offset` is 
```
[ -56    0    0  -95    0    0    0 -155    0    0    0    0    0    0
    0    0    0    0    0    0 -359    0    0    0 -423    0 -444    0
    0    0 -506    0    0    0    0 -585    0    0    0 -646    0    0
    0    0    0    0 -747    0    0    0    0    0    0    0    0    0
 -897    0    0    0    0    0    0    0    0    0    0    0   -1]
```