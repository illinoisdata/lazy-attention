"""What the packed block table can represent, in one place.

The decode kernels read `[physical_block_idx:32 | q_offset:16 | q_mask:16]`, so
q_offset has 16 bits. Whether a request fits has to be decided at admission --
by then the engine is committed -- while the packing itself happens in the model
runner, so the bound is defined here and both sides import it.
"""
from __future__ import annotations

from typing import Sequence

# Past this, the shifted offset carries into the physical-block field: the
# kernel reads a different (possibly out-of-range) block and de-rotates by a
# wrapped position, with nothing raised anywhere.
MAX_PACKED_Q_OFFSET = 0xFFFF


def max_rotation_offset(document_lens: Sequence[int],
                        document_lens_padded: Sequence[int]) -> int:
    """The largest +1-biased q_offset `metadata_for_lazy_attention` will emit.

    Document `d` rotates by the total padding plus the *true* lengths of the
    documents before it, so the largest is the one on the last document:

        max = sum(padded) - true[-1] + 1

    Note what this is not. It is not the size of the document region: a single
    block-aligned document has no padding and nothing before it, so its offset
    is 1 no matter how long it is. What it bounds is everything ahead of the
    last document.
    """
    if not document_lens:
        return 0
    total_padding = sum(
        padded - true for true, padded in zip(document_lens,
                                              document_lens_padded))
    return total_padding + sum(document_lens[:-1]) + 1
