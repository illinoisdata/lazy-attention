"""Block-Attention scheduler.

Reuses the lazy scheduler entirely (document merging + per-block q_offset =
abs_rot_pos + 1 + q_mask). The only addition is copy-on-write: after normal
scheduling, every document block of a freshly-scheduled lazy request is copied
to a fresh block, and the request's block table is redirected to the copy. The
runner then rotates the *copies* to their absolute positions, leaving the
canonical (offset-0) cached document blocks untouched -- so they can be re-used
by other requests / the prefix cache without corruption or rotation drift.

This is intentionally naive and slow (always copy, never share a rotated
block); it is a correctness baseline, not a performance path.

See [[block-attn-vllm-implementation]].
"""

from __future__ import annotations

from collections import deque

from vllm.logger import init_logger

from lazy.core.sched.scheduler import LazyScheduler

logger = init_logger(__name__)


class BlockAttnScheduler(LazyScheduler):

    def schedule(self):
        # Block-Attention rotates the SHARED document keys before the kernel,
        # which is only valid once the documents have actually been computed in
        # a PRIOR step. The lazy scheduler would otherwise prefill the documents
        # and the query in the SAME step (doc readiness = hash-registered at
        # allocation), where the document K does not exist yet at rotation time.
        # So here we defer any query whose documents are not yet computed: spawn
        # the document requests now and reconsider the query on a later step.
        deferred = self._defer_queries_with_uncomputed_docs()
        output = super().schedule()
        # Re-queue deferred queries for the next step (their docs are computed
        # by this step's forward).
        for req in reversed(deferred):
            self.waiting.appendleft(req)
        self._apply_copy_on_write(output)
        return output

    def _defer_queries_with_uncomputed_docs(self) -> list:
        kept: deque = deque()
        deferred: list = []
        docs_to_spawn: list = []
        for req in list(self.waiting):
            if (getattr(req, "has_documents", False)
                    and not getattr(req, "is_document_request", False)):
                flags = self.is_doc_ready(req)
                if not all(flags):
                    for doc_idx, ready in enumerate(flags):
                        if not ready:
                            docs_to_spawn.append((doc_idx, req))
                    deferred.append(req)
                    continue
            kept.append(req)

        # Document requests (and any non-deferred queries) stay; spawn the
        # missing document requests (added to the front of `waiting`) so they
        # are scheduled + computed this step. Deferred queries are held out of
        # `waiting` for this step and re-queued by the caller afterwards.
        self.waiting = kept
        for doc_idx, req in docs_to_spawn:
            self.add_doc_request(doc_idx, req)
        return deferred

    def _apply_copy_on_write(self, output) -> None:
        """Copy each document's matched blocks and record where to place them.

        We ask the block manager for the AUTHORITATIVE per-document matched
        block list (`get_computed_blocks_docs`) rather than inferring document
        blocks from the block-table order -- locality is not guaranteed, so the
        matched list is the source of truth for which physical block holds each
        document block. We then walk it document-by-document, block-by-block,
        allocate a fresh copy for each, redirect the request's block table to the
        copy, and record (new_id, old_id, from_start, to_start) for the runner:

          - from_start = local position of the block's first token during the
            document's own prefill (block_offset * block_size).
          - to_start   = target contiguous position of that token. The query
            keeps its padded position, so documents start after the total
            padding (`abs_rot_pos`); this makes the relative distances contiguous
            and identical to the Block-Attention reference (see
            metadata_for_lazy_attention).
        """
        block_pool = self.kv_cache_manager.block_pool
        req_to_blocks = self.kv_cache_manager.req_to_blocks
        block_size = self.block_size

        olds_to_release = []  # decref only after ALL allocations (avoid races)
        for new_req in output.scheduled_new_reqs:
            if not getattr(new_req, "is_lazy", False) or new_req.q_offset is None:
                continue
            request = self.requests[new_req.req_id]
            block_ids = new_req.block_ids
            req_blocks = req_to_blocks[new_req.req_id]

            # Authoritative per-document matched block list.
            computed_blocks_docs, _ = \
                self.kv_cache_manager.get_computed_blocks_docs(request)
            num_doc_blocks = sum(len(b) for b in computed_blocks_docs)
            if num_doc_blocks == 0:
                continue
            if block_pool.get_num_free_blocks() < num_doc_blocks:
                logger.warning(
                    "block-attn COW: not enough free blocks for request %s "
                    "(need %d); skipping rotation.",
                    new_req.req_id, num_doc_blocks)
                continue

            real = request.document_lens
            padded = request.document_lens_padded
            abs_rot_pos = sum(p - r for p, r in zip(padded, real))  # total padding

            new_blocks = iter(block_pool.get_new_blocks(num_doc_blocks))
            cow_blocks = []  # (new_id, old_id, from_start, to_start)
            cursor = 0
            for doc_idx, doc_blocks in enumerate(computed_blocks_docs):
                for block_offset, old_block in enumerate(doc_blocks):
                    # Content/structure check: the matched canonical block must
                    # line up with this slot of the request's block table.
                    assert (old_block.block_id == block_ids[cursor]
                            == req_blocks[cursor].block_id), (
                        "block-attn COW: matched document block does not align "
                        "with the request block table")
                    new_block = next(new_blocks)
                    from_start = block_offset * block_size
                    to_start = abs_rot_pos + block_offset * block_size
                    cow_blocks.append((new_block.block_id, old_block.block_id,
                                       from_start, to_start))
                    olds_to_release.append(req_blocks[cursor])
                    req_blocks[cursor] = new_block
                    block_ids[cursor] = new_block.block_id
                    cursor += 1
                abs_rot_pos += real[doc_idx]
            # Stash for the runner (dynamic attr; lazy's NewRequestData is a
            # plain dataclass, so this is fine).
            new_req.cow_blocks = cow_blocks

        for old_block in olds_to_release:
            old_block.decr_ref()
