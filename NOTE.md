# Scheduler

In `block_pool.py`, `get_new_blocks` and `touch` will increase the `ref_cnt` of a list of KV blocks.
In `kv_cache_manager.py`, the reference counter is updated in `allocate_slots`.

Allocate slots before we merge requests, it will simplify the logic of `cache_full_blocks`

## Appendix

We explain the code design here.

For each doc, we would align its size to the multiple of block size (e.g., 16 tokens), therefore each doc would logicially have real_doc + padding.

Then for each doc we would add the position id starting from 0 to the doc length before the doc is used in the lazy request. We impl this by feed the normal request as individual part (a non lazy request) then the system would automatically store it with local position.

For a lazy request which would rely on the doc processed, we would gather the calculated docs, the query part would get the position id with \sum doc_padded_len, then in the kernel we would recover the real relative distance.

in the prefilling kernel, the K would be rotate accoding to the docs, since docs have local distance, the content in the same doc just need to shift the same distance, like the first doc, it need to shift \sum padding_lenght, then second doc need to shift \sum padding_lenght + the real length of first doc.

in the decoding kernel, TBD