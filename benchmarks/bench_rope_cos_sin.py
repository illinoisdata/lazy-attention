"""LOAD vs COMPUTE cos/sin in the lazy decode kernel.

The kernel needs a cos/sin pair per rotation offset. It can read one row of the
model's `cos_sin_cache`, or evaluate the RoPE formula in-kernel
(`LAZY_DECODE_COMPUTE_COS_SIN=1`). This decides which is faster, and by how
much, instead of asserting it.

What is controlled for:

* **Only the constexpr changes.** Both variants launch the same kernel with the
  same tensors; `COMPUTE_COS_SIN` is the single difference, so the comparison
  cannot pick up a different input distribution.
* **Drift.** The two variants are interleaved inside every repetition, so clock
  and thermal drift lands on both. Reported numbers are medians over
  repetitions, with the full range, so a claim is only made when the ranges
  separate.
* **The variable that actually matters.** Q is re-rotated only when `q_offset`
  changes -- once per document, not per block -- so the cos/sin cost is
  amortised over a document's blocks. `--doc-lens` sweeps that density; a
  context of 8192 split into 128-token documents pays it 64x more often than
  the same context in one piece.
* **Compile-time effects.** `n_regs` / `n_spills` are read off each compiled
  kernel, since the standing explanation for the default is occupancy.
* **Which kernel, and which RoPE.** Both are constexpr specializations with
  their own register budgets, so both are swept: the mixed decode kernel and
  the lazy-only one `LAZY_FORCE_SPLIT_DECODE` selects, each with Llama-3
  smoothed RoPE and with plain RoPE (`ROPE_TYPE=0`, which compiles the
  smoothing out). The project's own checkpoints use both -- the 1B Block-FT
  model has no `rope_scaling` at all, the 8B one does.

Correctness is checked too: the two paths must agree, or the faster one is
irrelevant.

    python benchmarks/bench_rope_cos_sin.py               # full sweep
    python benchmarks/bench_rope_cos_sin.py --quick       # one shape
    python benchmarks/bench_rope_cos_sin.py --json out.json
    python benchmarks/bench_rope_cos_sin.py --e2e         # model-level A/B

What it found (RTX 5070 Ti, sm_120, torch 2.7.0+cu128 / triton 3.4.0): over the
144 sweep cases (1-32 seqs x both kernels x both RoPE types) 75 separated and
compute lost every one of them, median 1.13x; the other 69 are reported as
inconclusive, not as wins. Compute turns a profit only for large-batch decode
at head_size=128 (0.89x at 128 seqs), where the load path is itself
register-bound so the swap buys occupancy. Hence LOAD stays the default.
docs/design.md 4.3.1 has the register and PTX numbers behind that.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import triton

from vllm.model_executor.layers.rotary_embedding import (Llama3RotaryEmbedding,
                                                         RotaryEmbedding)

from lazy.attention.ops.models.llama_v1 import (
    kernel_paged_attention_2d_llama, kernel_paged_attention_2d_llama_lazy_only)
from lazy.core.sched.scheduler import metadata_for_lazy_attention
from lazy.model_executor.rope import rope_meta_from_layer

# Both decode kernels the runner can pick. They share a signature and a cos/sin
# path; the lazy-only one drops the non-lazy branch, which is what
# `LAZY_FORCE_SPLIT_DECODE=1` selects when a batch is split so that every
# sequence in this launch is lazy. Fewer live values means a different register
# budget, so its LOAD-vs-COMPUTE trade-off is a separate measurement.
KERNELS = {
    "mixed": kernel_paged_attention_2d_llama,
    "lazy_only": kernel_paged_attention_2d_llama_lazy_only,
}
ROPE_KINDS = ["llama3", "plain"]

BLOCK_SIZE = 16
DTYPE = torch.bfloat16
# Llama-3 defaults; the RoPE table is built from these so the in-kernel formula
# and the table are the same function of position.
ROPE_BASE = 500000.0
ROPE_MAX_POSITION = 131072
ROPE_SCALING = dict(scaling_factor=8.0, low_freq_factor=1.0,
                    high_freq_factor=4.0, orig_max_position=8192)


@dataclass
class Shape:
    name: str
    num_query_heads: int
    num_kv_heads: int
    head_size: int


# The decode kernel is launched per (sequence, kv head) and its tile is
# [max(next_pow2(query_heads // kv_heads), 16), head_size] -- so what a shape
# costs in registers is set by head_size and by the GQA group size *once the
# group exceeds 16*, not by parameter count. Llama-3 8B/70B/405B all land on
# head_size=128 and a group of 4/8/16, i.e. the same compiled kernel; `70B` is
# here to make that checkable rather than asserted. The last two are the shapes
# that do move the register pressure.
SHAPES = [
    Shape("1B", num_query_heads=32, num_kv_heads=8, head_size=64),
    Shape("8B", num_query_heads=32, num_kv_heads=8, head_size=128),
    Shape("70B", num_query_heads=64, num_kv_heads=8, head_size=128),
    Shape("hs256", num_query_heads=32, num_kv_heads=8, head_size=256),
    Shape("mqa32", num_query_heads=32, num_kv_heads=1, head_size=128),
]

# The full sweep stays on the two shapes the project actually runs; the rest are
# opt-in via --shapes.
DEFAULT_SHAPES = ["1B", "8B"]


# q_offset occupies 16 bits of the packed block table, so a rotation offset
# beyond this would run into the physical-block field.
MAX_PACKED_Q_OFFSET = 0xFFFF
# What the agreement check can and cannot see. The load path rounds cos/sin
# through a bf16 table, giving a relative error of ~2**-9 on the rotated Q;
# that propagates through the scores and softmax into an absolute output error
# of ~2e-3 for the N(0,1) inputs built here. So a sub-percent difference is
# indistinguishable from rounding no matter how it is measured -- verified by
# perturbing one path by 1% and watching it disappear into the noise. This
# bound is therefore set to catch *gross* divergence: a dropped rotation, a
# wrong block, a kernel that read the wrong offset field, all of which move
# outputs by O(1). It is not a precision claim.
#
# Absolute, not relative: an output element near zero is a near-cancellation of
# a sum over V, so its *relative* error is unbounded for reasons that have
# nothing to do with correctness, and dividing by the tensor's peak instead
# just makes the threshold depend on which random inputs were drawn.
MAX_ABS_DISAGREEMENT = 0.02
# Width of the measurement counter in the e2e query template; see
# `check_query_sets_are_equal_work`.
MEASUREMENT_DIGITS = 3


def padding_plan(num_docs: int, seq_idx: int) -> list[int]:
    """Per-document padding for one sequence: a fixed multiset, permuted.

    Padding is the mechanism the real metadata has for shifting rotation
    offsets -- a document's offset is the total padding plus the true lengths
    ahead of it -- so sequences have to differ in it, or the whole batch reads
    the same handful of `cos_sin_cache` rows and the load path gets an L2 hit
    rate no real batch would give it.

    They differ in *where* the padding sits, not in how much there is. Every
    sequence gets the same multiset in a different order, so all of them have
    the same total true context length, the same block layout, and the same
    number of rotations; only the offsets move. Scaling one pad value with
    `seq_idx` instead (the earlier version) changed the true length per
    sequence, and gave the `pad == 0` rows one rotation fewer than the rest --
    so a "single document" case was really a mix of zero- and one-rotation
    sequences, and the per-sequence work varied with the batch index.

    The first entry is non-zero, so the total padding is always at least 1 and
    the first document always rotates. With one document there is nothing to
    rotate between, and every sequence necessarily gets the same plan.

    `seq_idx` selects a permutation by factorial-base (Lehmer) index, which is
    distinct for `seq_idx < num_docs!`. Cyclically rotating the list instead
    (the previous version) has period `num_docs`, and since `base` itself
    repeats every `BLOCK_SIZE` entries, a period of only 16 for any longer
    document list -- so a 256-sequence batch had 16 distinct access traces and
    handed the load path a working set 16x smaller than the label claimed,
    biasing exactly the large-batch crossover this is used to measure.
    `run_case` reports the distinct-trace count so a regression here is visible
    rather than inferred.
    """
    items = [(doc_idx * 5 + 1) % BLOCK_SIZE for doc_idx in range(num_docs)]
    plan, index = [], seq_idx
    for slot in range(num_docs, 0, -1):
        index, pick = divmod(index, slot)
        plan.append(items.pop(pick))
    return plan


@dataclass
class Case:
    shape: Shape
    num_seqs: int
    context_len: int
    doc_len: int

    def validate(self) -> None:
        """Reject shapes the packed-table model cannot represent honestly.

        Each is a case where the benchmark would still run and still print a
        number, but the number would describe something other than the label.
        """
        if self.context_len % BLOCK_SIZE:
            raise ValueError(
                f"--context-lens must be a multiple of {BLOCK_SIZE}, got "
                f"{self.context_len}: the cache and the packed table would be "
                f"sized by flooring while the kernel walks ceil(seq_len / "
                f"{BLOCK_SIZE}) blocks, reading past both.")
        if self.doc_len % BLOCK_SIZE:
            raise ValueError(
                f"--doc-lens must be a multiple of {BLOCK_SIZE}, got "
                f"{self.doc_len}: documents are padded to whole blocks by the "
                f"scheduler, so an unaligned length would advance positions by "
                f"{self.doc_len} while changing offset every "
                f"{max(self.doc_len // BLOCK_SIZE, 1)} block(s) -- a "
                f"combination the real metadata never produces.")
        if self.context_len % self.doc_len:
            raise ValueError(
                f"--context-lens {self.context_len} is not a multiple of "
                f"--doc-lens {self.doc_len}: the last document would be a "
                f"partial one with a different rotation density, while the "
                f"reported document length and rotation count would describe "
                f"whole ones.")
        if self.max_q_offset > MAX_PACKED_Q_OFFSET:
            raise ValueError(
                f"context {self.context_len} with {self.doc_len}-token "
                f"documents needs a rotation offset of {self.max_q_offset}, "
                f"past the {MAX_PACKED_Q_OFFSET} the packed table's 16-bit "
                f"field holds; the excess bits would land in the block index.")

    @property
    def doc_blocks_per_seq(self) -> int:
        return self.context_len // BLOCK_SIZE

    @property
    def blocks_per_seq(self) -> int:
        """Document blocks plus the query/decode block.

        Production always walks one more block than the documents occupy: the
        partially filled one holding the query prompt and the tokens generated
        so far, whose metadata entry is the `q_offset = 1` reset. Dropping it
        (the earlier version did) skipped a block of the loop, a partial-block
        mask, and the one offset transition that costs nothing to serve.
        """
        return self.doc_blocks_per_seq + 1

    @property
    def seq_len(self) -> int:
        """Context plus the token being decoded, which lives in the last block."""
        return self.context_len + 1

    @property
    def num_docs(self) -> int:
        return max(self.context_len // self.doc_len, 1)

    @property
    def max_q_offset(self) -> int:
        """Largest +1-biased offset this case puts in the packed table.

        Mirrors `metadata_for_lazy_attention`: total padding, plus the true
        lengths of the documents ahead of the last one, over every padding
        plan `build_inputs` will use.
        """
        best = 0
        for seq_idx in range(self.num_seqs):
            pads = padding_plan(self.num_docs, seq_idx)
            true_lens = [self.doc_len - pad for pad in pads]
            best = max(best, sum(pads) + sum(true_lens[:-1]) + 1)
        return best


@dataclass
class Timing:
    per_iter_ms: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.per_iter_ms)

    @property
    def lo(self) -> float:
        return min(self.per_iter_ms)

    @property
    def hi(self) -> float:
        return max(self.per_iter_ms)


def build_inputs(case: Case, device: torch.device, rope_kind: str = "llama3"):
    """Decode-shaped inputs: one query token per sequence, KV already cached."""
    shape = case.shape
    num_seqs = case.num_seqs
    x = 16 // DTYPE.itemsize

    # Distinct blocks per sequence -- sharing them would hand both variants an
    # unrealistically warm L2 and flatten the very cost being measured.
    num_blocks = num_seqs * case.blocks_per_seq + 1
    key_cache = torch.randn(num_blocks,
                            shape.num_kv_heads,
                            shape.head_size // x,
                            BLOCK_SIZE,
                            x,
                            dtype=DTYPE,
                            device=device)
    value_cache = torch.randn(num_blocks,
                              shape.num_kv_heads,
                              shape.head_size,
                              BLOCK_SIZE,
                              dtype=DTYPE,
                              device=device)

    query = torch.randn(num_seqs,
                        shape.num_query_heads,
                        shape.head_size,
                        dtype=DTYPE,
                        device=device)
    output = torch.empty_like(query)

    seq_lens = torch.full((num_seqs, ), case.seq_len, dtype=torch.int32,
                          device=device)
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32,
                                   device=device)

    # Packed block table, exactly as LazyGPUModelRunner builds it:
    #   [physical_block_idx:32 | q_offset:16 | q_mask:16]
    block_ids = torch.arange(1, num_seqs * case.blocks_per_seq + 1,
                             dtype=torch.int64,
                             device=device).view(num_seqs,
                                                 case.blocks_per_seq)
    # Rotation metadata comes from the scheduler's own function rather than
    # being written out here, so the benchmark cannot time a layout the engine
    # would never produce. Sequences differ in their documents' *padding*,
    # which is the mechanism the real metadata has for shifting offsets: a
    # document's offset is the total padding plus the true lengths ahead of it,
    # so 16 tokens of padding moves every offset in that sequence, including
    # the first document's. Every sequence still has the same block layout and
    # the same number of rotations -- only the addresses differ, which is the
    # point. Broadcasting one row to the whole batch would have every sequence
    # reading the same handful of cos_sin_cache rows: an L2 hit rate the load
    # path would not get in a real batch, and the same reason the KV blocks
    # above are distinct per sequence.
    q_offset_rows, q_mask_rows = [], []
    for seq_idx in range(num_seqs):
        # Each pad stays under one block, so `doc_len` remains the padded
        # length and the block layout is identical across sequences.
        pads = padding_plan(case.num_docs, seq_idx)
        document_lens = [case.doc_len - pad for pad in pads]
        document_lens_padded = [case.doc_len] * case.num_docs
        row_offset, row_mask = metadata_for_lazy_attention(
            SimpleNamespace(document_lens=document_lens,
                            document_lens_padded=document_lens_padded),
            BLOCK_SIZE)
        # The whole row, trailing query/decode entry included.
        assert len(row_offset) == case.blocks_per_seq
        q_offset_rows.append(row_offset)
        q_mask_rows.append(row_mask)
    q_offset = torch.tensor(q_offset_rows, dtype=torch.int64, device=device)
    q_mask = torch.tensor(q_mask_rows, dtype=torch.int64, device=device)
    assert int(q_offset.max()) <= MAX_PACKED_Q_OFFSET  # Case.validate()
    packed = (block_ids << 32) | (q_offset << 16) | q_mask

    # Which RoPE the model uses is a compile-time branch in the kernel, not a
    # parameter: ROPE_TYPE=0 compiles the Llama-3 smoothing (a wavelength
    # comparison and two selects per element) out of the compute path
    # entirely, so it is the specialization with the best case for computing
    # in-kernel and has to be measured rather than extrapolated from Llama-3.
    rope_args = dict(head_size=shape.head_size,
                     rotary_dim=shape.head_size,
                     max_position_embeddings=ROPE_MAX_POSITION,
                     base=ROPE_BASE,
                     is_neox_style=True,
                     dtype=DTYPE)
    if rope_kind == "llama3":
        rope = Llama3RotaryEmbedding(**rope_args, **ROPE_SCALING).to(device)
    elif rope_kind == "plain":
        rope = RotaryEmbedding(**rope_args).to(device)
    else:
        raise ValueError(f"unknown rope kind {rope_kind!r}")

    return dict(
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=packed.contiguous(),
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=shape.head_size**-0.5,
        k_scale=torch.ones(1, dtype=torch.float32, device=device),
        v_scale=torch.ones(1, dtype=torch.float32, device=device),
        num_query_heads=shape.num_query_heads,
        num_queries_per_kv=shape.num_query_heads // shape.num_kv_heads,
        num_queries_per_kv_padded=max(
            triton.next_power_of_2(shape.num_query_heads //
                                   shape.num_kv_heads), 16),
        block_table_stride=packed.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        IN_PRECISION=None,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_SIZE=shape.head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(shape.head_size),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=query_start_loc,
        rotary_dim=shape.head_size,
        rotary_dim_pow2=triton.next_power_of_2(shape.head_size),
        is_neox_style=True,
        is_lazy_ptr=torch.ones(num_seqs, dtype=torch.bool, device=device),
        q_offset_ptr=q_offset.to(torch.int32),
        q_mask_ptr=q_mask.to(torch.int32),
        cos_sin_cache_ptr=rope.cos_sin_cache,
        IGNORE_Q_MASK=False,
        **rope_meta_from_layer(rope),
    )


def launch(kwargs, compute_cos_sin: bool, num_seqs: int, num_kv_heads: int,
           kernel):
    return kernel[(num_seqs, num_kv_heads)](**kwargs,
                                            COMPUTE_COS_SIN=compute_cos_sin)


def kernel_metadata(compiled, case: Case, label: str) -> tuple[int, int]:
    """`n_regs` / `n_spills` off a compiled kernel, or a loud failure.

    On the pinned Triton (3.4) an ordinary `kernel[grid](...)` launch returns
    the `CompiledKernel`, so the warmup launches already hand back everything
    needed -- checked on this environment, where the load path reports 208
    registers and 0 spills on the 8B shape. Should a future version return
    something else, reading these with a `None` default would print `None` in
    the register line and quietly withdraw the evidence for the occupancy
    argument this whole benchmark rests on, while still printing timings that
    look complete. So it raises instead: use `kernel.warmup(...)` to obtain the
    compiled object explicitly if that day comes.
    """
    n_regs = getattr(compiled, "n_regs", None)
    n_spills = getattr(compiled, "n_spills", None)
    if n_regs is None or n_spills is None:
        raise RuntimeError(
            f"launching {case.shape.name}/{label} returned "
            f"{type(compiled).__name__}, which carries no n_regs/n_spills; "
            f"the register and occupancy numbers this benchmark reports come "
            f"from there, so it stops rather than reporting them as None "
            f"(triton {triton.__version__})")
    return n_regs, n_spills


def time_variant(kwargs, compute_cos_sin, case, iters, kernel) -> float:
    """Mean ms per launch over `iters`, measured with CUDA events."""
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch(kwargs, compute_cos_sin, case.num_seqs,
               case.shape.num_kv_heads, kernel)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(case: Case, device, iters: int, reps: int, warmup: int,
             kernel_name: str = "mixed", rope_kind: str = "llama3") -> dict:
    kwargs = build_inputs(case, device, rope_kind)
    kernel = KERNELS[kernel_name]

    # Compile + warm caches for both variants before timing either.
    compiled = {}
    for compute in (False, True):
        for _ in range(warmup):
            compiled[compute] = launch(kwargs, compute, case.num_seqs,
                                       case.shape.num_kv_heads, kernel)
    torch.cuda.synchronize()

    # Agreement: same inputs, same kernel, different cos/sin source.
    outputs = {}
    for compute in (False, True):
        kwargs["output_ptr"].zero_()
        launch(kwargs, compute, case.num_seqs, case.shape.num_kv_heads, kernel)
        torch.cuda.synchronize()
        outputs[compute] = kwargs["output_ptr"].clone().float()
    diff = (outputs[False] - outputs[True]).abs()
    magnitude = torch.maximum(outputs[False].abs(), outputs[True].abs())
    scale = outputs[False].abs().max().clamp(min=1e-6)
    # A timing comparison between two paths that disagree is meaningless, so
    # this fails the run rather than printing the discrepancy underneath a
    # performance conclusion. The bound is bf16's own resolution: the two paths
    # are not expected to be bit-identical (one rounds through a bf16 table),
    # only to agree to within it.
    # Checked before the tolerance: a NaN anywhere makes `rel_diff` NaN, and
    # `nan > tol` is False -- the comparison below would wave through exactly
    # the failure it exists to catch.
    for name, output in (("load", outputs[False]), ("compute", outputs[True])):
        if not torch.isfinite(output).all():
            raise AssertionError(
                f"the {name} path produced non-finite output on "
                f"{case.shape.name} seqs={case.num_seqs} "
                f"ctx={case.context_len} doc={case.doc_len} "
                f"({int((~torch.isfinite(output)).sum())} of "
                f"{output.numel()} values) -- a correctness failure, not a "
                f"benchmark result.")
    gross = diff > MAX_ABS_DISAGREEMENT
    if gross.any():
        raise AssertionError(
            f"load and compute disagree grossly on {case.shape.name} "
            f"seqs={case.num_seqs} ctx={case.context_len} doc={case.doc_len}: "
            f"{int(gross.sum())} of {diff.numel()} elements past "
            f"{MAX_ABS_DISAGREEMENT} absolute (worst {diff.max().item():.3e}, "
            f"bf16 cos/sin rounding lands near 2e-3) -- this is a correctness "
            f"regression, not a benchmark result.")

    timings = {False: Timing(), True: Timing()}
    for rep in range(reps):
        # Interleaved, so drift hits both variants inside one repetition, and
        # order-alternated, so whichever runs first does not keep the colder
        # caches every time.
        order = (False, True) if rep % 2 == 0 else (True, False)
        for compute in order:
            timings[compute].per_iter_ms.append(
                time_variant(kwargs, compute, case, iters, kernel))

    # How many cos/sin pairs a sequence actually evaluates, counted off the
    # metadata that was generated rather than predicted from the case: the
    # kernel fetches one whenever the offset changes to a non-sentinel value.
    offsets = kwargs["q_offset_ptr"]
    changes = (offsets[:, 1:] != offsets[:, :-1]) & (offsets[:, 1:] > 1)
    rotations = changes.sum(dim=1) + (offsets[:, 0] > 1)
    mean_rotations = rotations.float().mean().item()
    # How many distinct cos/sin access traces the batch actually contains. Two
    # sequences with the same offset row read the same table rows, so a batch
    # with few distinct rows gives the load path an L2 working set that shrinks
    # with the repetition rather than with the batch -- the confound the
    # distinct KV blocks and the per-sequence padding plans exist to avoid.
    distinct_traces = len({tuple(row) for row in offsets.tolist()})

    load_regs, load_spills = kernel_metadata(compiled[False], case, "load")
    compute_regs, compute_spills = kernel_metadata(compiled[True], case,
                                                   "compute")

    load, comp = timings[False], timings[True]
    result = dict(
        shape=case.shape.name,
        kernel=kernel_name,
        rope=rope_kind,
        num_seqs=case.num_seqs,
        context_len=case.context_len,
        doc_len=case.doc_len,
        rotations_per_seq=mean_rotations,
        distinct_traces=distinct_traces,
        load_ms=load.median,
        load_range=[load.lo, load.hi],
        compute_ms=comp.median,
        compute_range=[comp.lo, comp.hi],
        ratio=comp.median / load.median,
        separated=load.hi < comp.lo or comp.hi < load.lo,
        max_abs_diff=diff.max().item(),
        max_rel_diff=(diff.max() / scale).item(),
        regs={"load": load_regs, "compute": compute_regs},
        spills={"load": load_spills, "compute": compute_spills},
    )
    del kwargs, outputs
    torch.cuda.empty_cache()
    return result


def run_e2e(model: str, rounds: int, max_tokens: int, num_prompts: int,
            max_num_seqs: int, gpu_util: float) -> int:
    """What the kernel difference is worth end to end, on a real model.

    Both variants run against one engine in one process -- the switch is read
    per call -- so weights, cache state and sampling are identical and only the
    kernel constexpr differs. Rounds alternate.
    """
    import os
    import time

    # Must be in-process. The A/B flips LAZY_DECODE_COMPUTE_COS_SIN between
    # rounds and the kernel reads it per call; with a worker process the engine
    # forks before those mutations and both labelled variants would silently
    # run whichever value the worker inherited.
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") not in (None, "0"):
        print("note: forcing VLLM_ENABLE_V1_MULTIPROCESSING=0 -- the per-round "
              "switch does not reach a worker process")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import lazy.__vllm__  # noqa: F401  patches vLLM
    from vllm import LLM, SamplingParams

    documents = [[
        f"Document {doc_idx}: " + "context words for retrieval augmented "
        "generation " * 12 for doc_idx in range(8)
    ] for _ in range(num_prompts)]
    # Every measurement gets its own query set, for two reasons. Repeating a
    # query set makes the *second* variant of a round run against a prefix cache
    # the first one just filled -- the single largest confound here, worth more
    # than 2x in throughput. And identical queries within a set make every
    # request after the first a full prefix-cache hit (documents and query
    # alike), which trips `assert num_new_tokens > 0` in the lazy scheduler.
    # The documents are deliberately *not* varied: reusing a hot corpus across
    # requests is the case LazyAttention exists for.
    def query_set(measurement: int) -> list[str]:
        # Numbered by measurement, never by variant name: "load" and "compute"
        # can tokenize to different lengths, which would hand the two sides
        # different prefill and decode work on some tokenizers. The number is
        # zero-padded to a fixed width for the same reason -- "10" is two
        # tokens where "9" is one on some vocabularies, so an unpadded counter
        # reintroduces the imbalance it was meant to remove. Verified against
        # the real tokenizer below rather than assumed.
        return [
            f"Question {measurement:0{MEASUREMENT_DIGITS}d}-{idx}: what do the "
            f"documents describe? Answer:" for idx in range(num_prompts)
        ]

    def check_query_sets_are_equal_work(llm, measurements: range) -> None:
        """Every measurement must be the same amount of tokenized work.

        The A/B assigns measurements to variants alternately, so a query set
        that tokenizes longer than another does not average out -- it lands on
        whichever variant drew it, as a throughput difference with the same
        sign and the same shape as the effect being measured.
        """
        tokenizer = llm.get_tokenizer()
        lengths = {
            measurement: [len(ids) for ids in
                          tokenizer(query_set(measurement))["input_ids"]]
            for measurement in measurements
        }
        reference = lengths[measurements[0]]
        for measurement, observed in lengths.items():
            if observed != reference:
                mismatched = next(idx for idx, (a, b) in
                                  enumerate(zip(observed, reference)) if a != b)
                raise ValueError(
                    f"query set {measurement} does not tokenize to the same "
                    f"lengths as set {measurements[0]} (prompt {mismatched}: "
                    f"{observed[mismatched]} vs {reference[mismatched]} "
                    f"tokens). The A/B would compare unequal work; widen "
                    f"MEASUREMENT_DIGITS or reword the query template.")

    # max_num_seqs is the variable under test: the kernel-level win only
    # appears once the decode batch is large, so it has to be settable.
    llm = LLM(model=model, gpu_memory_utilization=gpu_util,
              max_model_len=4096, max_num_seqs=max_num_seqs,
              enforce_eager=True)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0,
                              ignore_eos=True)
    # +1 for the warmup round; both variants run in every round.
    check_query_sets_are_equal_work(llm, range(1, 2 * (rounds + 1) + 1))

    def one_round(measurement: int) -> tuple[float, int]:
        start = time.perf_counter()
        outputs = llm.generate(prompts=query_set(measurement),
                               sampling_params=sampling,
                               document_seqs=documents, use_tqdm=False)
        elapsed = time.perf_counter() - start
        return elapsed, sum(len(o.outputs[0].token_ids) for o in outputs)

    results: dict[str, list[float]] = {"load": [], "compute": []}
    measurement = 0
    for round_idx in range(rounds + 1):
        for name in (("load", "compute")
                     if round_idx % 2 == 0 else ("compute", "load")):
            os.environ["LAZY_DECODE_COMPUTE_COS_SIN"] = (
                "1" if name == "compute" else "0")
            measurement += 1
            elapsed, tokens = one_round(measurement)
            if round_idx == 0:
                continue  # warmup round: JIT compile + cache warm
            results[name].append(tokens / elapsed)
    os.environ.pop("LAZY_DECODE_COMPUTE_COS_SIN", None)

    concurrency = min(num_prompts, max_num_seqs)
    print(f"\nend to end ({model}, {max_tokens} tokens x {num_prompts} "
          f"prompts, max_num_seqs={max_num_seqs} -> at most {concurrency} "
          f"concurrent, {rounds} rounds)")
    if concurrency < max_num_seqs:
        print(f"  note: only {num_prompts} prompts were submitted, so the "
              f"{max_num_seqs}-sequence regime this switch is measured in was "
              f"never reached -- raise --e2e-prompts")
    for name, values in results.items():
        print(f"  {name:>8}: {statistics.median(values):8.1f} tok/s   "
              f"(range {min(values):.1f} - {max(values):.1f})")
    load_median = statistics.median(results["load"])
    compute_median = statistics.median(results["compute"])
    print(f"  compute throughput is {(compute_median / load_median - 1) * 100:+.1f}%"
          " vs load")
    if min(len(values) for values in results.values()) < 3:
        # The kernel sweep refuses to call an overlapping pair, and the e2e
        # difference is smaller and noisier than the kernel one -- so a
        # percentage off one or two rounds per variant is a number, not a
        # result, and gets labelled as such rather than left to be quoted.
        print("  note: fewer than 3 rounds per variant -- that percentage is a "
              "single draw from a distribution wider than the effect; raise "
              "--e2e-rounds before quoting it")
    return 0


def positive_int(value: str) -> int:
    """A count that must actually count something.

    Zero is accepted by `int` and breaks a different part of the run in each
    case -- `--iters 0` divides by zero in `time_variant`, `--reps 0` leaves the
    timing lists empty for `statistics.median`, `--warmup 0` leaves no compiled
    kernel to read registers off, `--e2e-rounds 0` runs only the warmup round --
    each after the inputs are allocated and the kernels compiled. Rejected at
    parse time instead.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {parsed}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seqs", type=positive_int, nargs="+",
                        default=[1, 8, 32])
    parser.add_argument("--context-lens", type=positive_int, nargs="+",
                        default=[1024, 4096, 16384])
    parser.add_argument("--doc-lens", type=positive_int, nargs="+",
                        default=[128, 1024])
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES,
                        choices=[shape.name for shape in SHAPES])
    parser.add_argument("--rope", nargs="+", default=ROPE_KINDS,
                        choices=ROPE_KINDS,
                        help="RoPE specialization: llama3 smoothing, or plain "
                             "(ROPE_TYPE=0, which compiles the smoothing out)")
    parser.add_argument("--kernels", nargs="+", default=list(KERNELS),
                        choices=list(KERNELS),
                        help="decode kernel: the mixed one, or the lazy-only "
                             "one LAZY_FORCE_SPLIT_DECODE selects")
    parser.add_argument("--iters", type=positive_int, default=50)
    parser.add_argument("--reps", type=positive_int, default=7)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--quick", action="store_true",
                        help="one representative case")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--e2e", action="store_true",
                        help="model-level A/B instead of the kernel sweep")
    parser.add_argument("--e2e-model", default="hxia7/Llama-3.2-1B-Block-FT")
    parser.add_argument("--e2e-rounds", type=positive_int, default=4)
    parser.add_argument("--e2e-max-tokens", type=positive_int, default=128)
    # Defaulted to the concurrency limit: a smaller number silently
    # measures a small-batch run under a large-batch label.
    parser.add_argument("--e2e-prompts", type=positive_int, default=256)
    parser.add_argument("--e2e-max-num-seqs", type=positive_int, default=256)
    parser.add_argument("--e2e-gpu-util", type=float, default=0.6)
    args = parser.parse_args()

    if args.e2e:
        return run_e2e(args.e2e_model, args.e2e_rounds, args.e2e_max_tokens,
                       args.e2e_prompts, args.e2e_max_num_seqs,
                       args.e2e_gpu_util)

    if args.quick:
        args.shapes = ["8B"]
        args.num_seqs = [8]
        args.context_lens = [4096]
        args.doc_lens = [128]
        args.rope = ["llama3"]
        args.kernels = ["mixed"]

    device = torch.device("cuda")
    print(f"device : {torch.cuda.get_device_name(device)}")
    print(f"torch  : {torch.__version__}   triton: {triton.__version__}")
    print(f"timing : {args.iters} launches x {args.reps} interleaved reps\n")

    shapes = [shape for shape in SHAPES if shape.name in args.shapes]
    cases = [
        Case(shape, num_seqs, context_len, doc_len)
        for shape, num_seqs, context_len, doc_len in itertools.product(
            shapes, args.num_seqs, args.context_lens, args.doc_lens)
        if doc_len <= context_len
    ]
    for case in cases:
        case.validate()

    header = (f"{'shape':>5} {'kernel':>9} {'rope':>6} {'seqs':>5} {'ctx':>6} "
              f"{'doc':>5} {'rot':>5} {'load ms':>9} {'compute ms':>11} "
              f"{'ratio':>7} {'sep':>4} {'maxdiff':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for kernel_name, rope_kind, case in itertools.product(
            args.kernels, args.rope, cases):
        result = run_case(case, device, args.iters, args.reps, args.warmup,
                          kernel_name, rope_kind)
        results.append(result)
        print(f"{result['shape']:>5} {result['kernel']:>9} "
              f"{result['rope']:>6} {result['num_seqs']:>5} "
              f"{result['context_len']:>6} {result['doc_len']:>5} "
              f"{result['rotations_per_seq']:>5.1f} "
              f"{result['load_ms']:>9.4f} {result['compute_ms']:>11.4f} "
              f"{result['ratio']:>7.3f} "
              f"{'yes' if result['separated'] else 'no':>4} "
              f"{result['max_abs_diff']:>9.2e}")

    # Only cases whose timing ranges separate get a verdict. Counting a case
    # whose ranges overlap as a win for whichever median came out lower is how
    # noise gets reported as a result -- the rule this benchmark states in its
    # docstring, applied to its own summary.
    #
    # Summarised per (kernel, RoPE) rather than pooled: those are different
    # compiled kernels with different register budgets, so a pooled median
    # would average over the very thing that decides the answer.
    def summarize(label: str, subset: list[dict]) -> None:
        decided = [result for result in subset if result["separated"]]
        undecided = len(subset) - len(decided)
        if decided:
            ratios = [result["ratio"] for result in decided]
            print(f"\n{label}: compute/load over the {len(decided)} decided "
                  f"case(s): median {statistics.median(ratios):.3f}, min "
                  f"{min(ratios):.3f}, max {max(ratios):.3f}")
            print(f"  compute wins {sum(1 for r in ratios if r < 1)}, "
                  f"load wins {sum(1 for r in ratios if r > 1)}")
        else:
            print(f"\n{label}: no case separated, nothing conclusive")
        if undecided:
            print(f"  inconclusive (ranges overlap): "
                  f"{undecided}/{len(subset)}")

    # A case whose sequences share offset rows is measuring a smaller working
    # set than its batch size suggests. It is bounded by num_docs! and so is
    # unavoidable for short document lists -- reported rather than hidden.
    repeated = [result for result in results
                if result["distinct_traces"] < result["num_seqs"]]
    if repeated:
        worst = min(repeated,
                    key=lambda r: r["distinct_traces"] / r["num_seqs"])
        print(f"\nnote: {len(repeated)}/{len(results)} case(s) have fewer "
              f"distinct rotation traces than sequences (worst: "
              f"{worst['distinct_traces']} traces over {worst['num_seqs']} "
              f"seqs at ctx={worst['context_len']}/doc={worst['doc_len']}); "
              f"with few documents there are only so many orderings, and the "
              f"load path sees a warmer table there")

    for kernel_name, rope_kind in itertools.product(args.kernels, args.rope):
        subset = [result for result in results
                  if result["kernel"] == kernel_name
                  and result["rope"] == rope_kind]
        summarize(f"{kernel_name}/{rope_kind}", subset)
    if len(args.kernels) * len(args.rope) > 1:
        summarize("all configurations", results)
    # Registers are a property of the compiled kernel -- of the shape and the
    # constexpr specialization, not of the batch -- so one line per
    # (shape, kernel, rope) rather than one for the run.
    print()
    seen = set()
    for result in results:
        key = (result["shape"], result["kernel"], result["rope"])
        if key in seen:
            continue
        seen.add(key)
        regs, spills = result["regs"], result["spills"]
        print(f"{result['shape']:>5} {result['kernel']:>9} "
              f"{result['rope']:>6} registers/thread: load {regs['load']}, "
              f"compute {regs['compute']}   spills: load {spills['load']}, "
              f"compute {spills['compute']}")
    worst = max(results, key=lambda result: result["max_rel_diff"])
    print(f"worst |load - compute|: {worst['max_abs_diff']:.3e} abs, "
          f"{worst['max_rel_diff']:.3e} relative to peak output "
          f"(bfloat16 has ~8 mantissa bits, i.e. ~4e-3)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
