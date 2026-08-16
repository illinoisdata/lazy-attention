"""What the packed block table's 16-bit q_offset field can hold.

The bound is easy to state wrongly. It is *not* "documents under 64k tokens":
document `d` rotates by the total padding plus the true lengths of the documents
before it, so a single block-aligned document of any size is offset 1, while
many small ones add up.
"""
import pytest

from conftest import make_lazy_request
from lazy.core.sched.scheduler import metadata_for_lazy_attention
from lazy.engine.processor import _validate_rotation_offsets
from lazy.utils.rotation import MAX_PACKED_Q_OFFSET, max_rotation_offset

BLOCK_SIZE = 16


@pytest.mark.unit
@pytest.mark.parametrize("document_lens", [
    [6],
    [16, 16],
    [6, 15, 8, 2],
    [1, 1, 1, 1, 1],
    [4096, 17],
    [100] * 32,
])
def test_helper_agrees_with_the_scheduler(document_lens):
    """The bound is computed in two places; they must not drift apart.

    `max_rotation_offset` predicts what `metadata_for_lazy_attention` emits,
    from the lengths alone, because admission has to decide before a request
    object exists.
    """
    padded = [((length + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
              for length in document_lens]
    request = make_lazy_request(
        documents_token_ids_padded=[[0] * length for length in padded],
        document_lens=document_lens,
        document_lens_padded=padded,
    )

    q_offset, _ = metadata_for_lazy_attention(request, BLOCK_SIZE)

    assert max_rotation_offset(document_lens, padded) == max(q_offset)


@pytest.mark.unit
def test_one_huge_document_is_offset_one():
    """The limit is not a cap on document size, and must not be applied as one."""
    huge = MAX_PACKED_Q_OFFSET * 4

    assert max_rotation_offset([huge], [huge]) == 1
    _validate_rotation_offsets([huge], [huge])  # accepted


@pytest.mark.unit
def test_many_documents_reach_the_limit():
    lens = [1024] * 64  # 65536 tokens ahead of the last one
    padded = list(lens)

    assert max_rotation_offset(lens, padded) == 1024 * 63 + 1

    with pytest.raises(ValueError, match="past the"):
        _validate_rotation_offsets([1024] * 65, [1024] * 65)


@pytest.mark.unit
def test_padding_counts_toward_the_offset():
    """Every document's padding rotates every later document, including the last."""
    # One token per document, each padded to a whole block: 15 tokens of
    # padding each, and the padding of *all* of them lands in every offset.
    lens = [1] * 8
    padded = [BLOCK_SIZE] * 8

    assert max_rotation_offset(lens, padded) == 8 * 15 + 7 + 1
