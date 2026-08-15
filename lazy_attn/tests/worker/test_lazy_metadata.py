"""Each attention layer must receive its own KV cache group's block table.

Block ids are only meaningful inside the group that allocated them, so a layer
in group 1 handed group 0's packed table would read another group's cache --
silently, since the ids are valid integers either way.

The attach step is exercised directly on a stand-in `self`: it only reads the
runner's buffers, so this stays CPU-only and does not need an engine.
"""
from types import SimpleNamespace

import pytest
import torch

from lazy.worker.gpu_model_runner import LazyGPUModelRunner

NUM_REQS = 2
NUM_BLOCKS = 4
GROUP_MARKER = (1000, 2000)  # so a wrong table is obvious, not just unequal


def make_runner_stub(num_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
        is_lazy_req=torch.ones(NUM_REQS, dtype=torch.bool),
        lazy_variant=torch.ones(NUM_REQS, dtype=torch.int32),
        lazy_offset=torch.zeros((NUM_REQS, NUM_BLOCKS), dtype=torch.int32),
        lazy_mask=torch.zeros((NUM_REQS, NUM_BLOCKS), dtype=torch.int32),
        packed_block_tables=[
            torch.full((NUM_REQS, NUM_BLOCKS),
                       GROUP_MARKER[group_idx],
                       dtype=torch.int64) for group_idx in range(num_groups)
        ],
        _layer_to_kv_group={"layer.0": 0, "layer.1": 1},
    )


@pytest.mark.unit
def test_each_layer_gets_its_own_group_table():
    runner = make_runner_stub(num_groups=2)
    metadata = {
        "layer.0": SimpleNamespace(),
        "layer.1": SimpleNamespace(),
    }

    LazyGPUModelRunner._attach_lazy_attn_metadata(runner, metadata, NUM_REQS)

    assert int(metadata["layer.0"].packed_block_table[0, 0]) == GROUP_MARKER[0]
    assert int(metadata["layer.1"].packed_block_table[0, 0]) == GROUP_MARKER[1]


@pytest.mark.unit
def test_rotation_tensors_are_shared_across_layers():
    """Only the block table is group-specific; the rotation metadata is not."""
    runner = make_runner_stub(num_groups=2)
    metadata = {
        "layer.0": SimpleNamespace(),
        "layer.1": SimpleNamespace(),
    }

    LazyGPUModelRunner._attach_lazy_attn_metadata(runner, metadata, NUM_REQS)

    for name in ("is_lazy", "lazy_variant", "q_offset", "q_mask"):
        assert getattr(metadata["layer.0"], name).data_ptr() == getattr(
            metadata["layer.1"], name).data_ptr()


@pytest.mark.unit
def test_unknown_layer_falls_back_to_the_first_group():
    """A metadata object with no layer name (older single-object shape) still
    gets a table rather than raising."""
    runner = make_runner_stub(num_groups=1)
    metadata = SimpleNamespace()

    LazyGPUModelRunner._attach_lazy_attn_metadata(runner, metadata, NUM_REQS)

    assert int(metadata.packed_block_table[0, 0]) == GROUP_MARKER[0]
    assert metadata.packed_block_table.shape[0] == NUM_REQS
