# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Early remote output from the last head group of dense, noncausal attention."""

import math

import torch

from .ce_plan import HeadPlan


def eligible(kernel, metadata, distributed_rows):
    owner = getattr(kernel, "__self__", None)
    if getattr(owner, "causal", None) is not False or not hasattr(owner, "effective_attention"):
        return False
    _, backend, _ = owner.effective_attention()
    if backend is None or backend.get_name() != "FLASH_ATTN":
        return False
    if metadata is None:
        return True
    layout = metadata.video_layout
    if layout is None:
        if metadata.extra:
            return False
    else:
        # Dense FA3 ignores grid geometry. Only admit a single fully occupied
        # video grid; masks, packed rows and sparse backends keep their routes.
        grid = metadata.extra.get("vsa_dit_seq_shape")
        if (
            layout.prefix_len != 0
            or layout.video_spans
            or layout.used_len != distributed_rows
            or layout.latent_grid is None
            or math.prod(layout.latent_grid) != distributed_rows
            or len(metadata.extra) != 1
            or type(grid) is not tuple
            or any(type(dim) is not int for dim in grid)
            or grid != layout.latent_grid
            or metadata.joint_query is not None
        ):
            return False
    return (
        (metadata.joint_query is None or metadata.joint_query.shape[1] == 0)
        and metadata.attn_mask is None
        and metadata.joint_attn_mask is None
        and metadata.full_attn_spans is None
        and metadata.query_ranges is None
        and metadata.packed_padding is None
        and metadata.cross_query_mask is None
    )


def candidate(heads, rows, world, sms):
    """Propose the smallest leading head group and a useful local Q tail.

    The 128-row SM90 wave estimate just bounds the real-prefix candidates.
    Prefer the full local return. A half tail is useful near a wave boundary.
    """

    def waves(length, width):
        return math.ceil(math.ceil(length / 128) * width / sms)

    local = rows // world
    head_only = ((heads,),)
    for first in range(1, heads // 2 + 1):
        last = heads - first
        if waves(rows, first) + waves(rows, last) > waves(rows, heads):
            continue
        choices = ((heads,), (first, last))
        if len(head_only) == 1:
            head_only = choices
        for tail in (local, local // 2):
            if tail and waves(rows - tail, last) + waves(tail, last) <= waves(rows, last):
                return choices, tail
    # A useful split can leave only one wave in its second group. In that
    # case ordinary head overlap can still win the real-layer comparison.
    return head_only, 0


def query_row(source, owner, world):
    if source == owner:
        return world - 1
    return owner if source == world - 1 else source


class QueryTailPipeline(HeadPlan):
    """Return remote O of the last head group while computing its local query rows."""

    def __init__(self, *args, tail_rows, **kwargs):
        super().__init__(*args, **kwargs)
        if self.chunks != 2 or self.joint_len or not 0 < tail_rows <= self.local_seq:
            raise ValueError("The query-tail control requires two groups and distributed queries only")
        self.tail_rows = tail_rows
        q, k, v = self.parts[-1]
        self.tail_inputs = ((q[:, :-tail_rows], k, v), (q[:, -tail_rows:], k, v))
        self.local_batches = ()
        if q.element_size() == 2:
            from .local_copy import LocalCopyBatch

            self.local_batches = tuple(LocalCopyBatch.prepare(ops) for ops in self.local_forward)
            for batch in self.local_batches:
                batch.load()
        # One persistent descriptor per local slice, with exactly the same
        # total width and row count as the original local return.
        self.local_returns = []
        head_bytes = self.dim * q.element_size()
        width = self.head_sizes[-1] * head_bytes
        for dest_start, rows in ((0, self.local_seq - tail_rows), (self.local_seq - tail_rows, tail_rows)):
            if rows:
                self.local_returns.append((dest_start, self.geometry(0, 0, 0, 0, width, rows)))

    def _forward_geometry(self, group):
        ops = super()._forward_geometry(group)
        if group == self.chunks - 1:
            # Only the Q rows move. Keep K/V in canonical token order, so the
            # softmax reductions traverse exactly the original key sequence.
            for owner, op in enumerate(ops[: self.world]):
                dest_row = query_row(self.rank, owner, self.world)
                op[0] += (dest_row - self.rank) * self.local_seq * op[1]
        return ops

    def reverse_operations(self, group, out, peer):
        ops = super().reverse_operations(group, out, peer)
        if group == self.chunks - 1 and peer != self.rank:
            row = query_row(peer, self.rank, self.world)
            ops[0][2] += (row - peer) * self.local_seq * ops[0][3]
        return ops

    def run(self):
        compute, comm = torch.cuda.current_stream(), self.stream
        self.enter.record(compute)
        with torch.cuda.stream(comm):
            comm.wait_event(self.enter)
            self.submit_forward(0)
        if self.local_batches:
            for batch in self.local_batches:
                batch.submit()
        else:
            self.emit_batch_on(self.local_forward_all, compute)
        compute.wait_event(self.ready[0])
        out0 = self.attend(*self.parts[0])
        self.produced[0].record(compute)
        with torch.cuda.stream(comm):
            self.submit_forward(1)
            comm.wait_event(self.produced[0])
            self.submit_reverse(0, out0)
        self.emit_batch_on(self.reverse_operations(0, out0, self.rank), compute)
        compute.wait_event(self.ready[1])
        remote = self.attend(*self.tail_inputs[0])
        self.admitted[0].record(compute)
        with torch.cuda.stream(comm):
            comm.wait_event(self.admitted[0])
            self.submit_reverse(1, remote)
            # Remote O is complete at this fence. The remaining FA reads only
            # input group 1. Before any peer can reuse that group, the next
            # call's F0 barrier requires every rank to finish its local tail.
            # Group 0 has separate storage. Local writes on the next call
            # follow this rank's consumer on the owning compute stream.
            self.output.handle.barrier()
            self.done.record(comm)
        local = self.attend(*self.tail_inputs[1])
        dest = self.output.handle.buffer_ptrs[self.rank]
        dest_pitch = self.output.tensor.stride(1) * local.element_size()
        head = self.head_offsets[-1] + self.rank * self.head_sizes[-1]
        for dest_start, op in self.local_returns:
            # The optional leading local fragment follows all remote rows.
            is_tail = dest_start == self.local_seq - self.tail_rows
            src = local if is_tail else remote
            source_row = 0 if is_tail else (self.world - 1) * self.local_seq
            op[0] = dest + dest_start * dest_pitch + head * self.dim * local.element_size()
            op[1] = dest_pitch
            op[3] = src.stride(1) * src.element_size()
            op[2] = src.data_ptr() + source_row * op[3]
            self.emit_on(op, compute)
        compute.wait_event(self.done)
        return self.output.tensor
