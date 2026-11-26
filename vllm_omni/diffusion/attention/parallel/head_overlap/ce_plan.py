# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Copy-engine plan for one layout. All ranks use the same head order and peer fences."""

import ctypes
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from vllm_omni.diffusion.distributed.a2a_permute import _ce_runtime

_COPY_SUBMITTER = os.environ.get("VLLM_OMNI_HEAD_OVERLAP_COPY_SUBMITTER")


class SymmetricBuffer:
    def __init__(self, shape, dtype, device, group):
        stride = torch.empty(shape, device="meta").stride()
        self.tensor = symm._SymmetricMemory.empty_strided_p2p(shape, stride, dtype, device)
        self.handle = symm.rendezvous(self.tensor, group.group_name)
        assert self.tensor.data_ptr() == self.handle.buffer_ptrs[dist.get_rank(group)]


class HeadPlan:
    @staticmethod
    def geometry(dest, dpitch, source, spitch, width, rows):
        values = [dest, dpitch, source, spitch, width, rows, 4]
        return (ctypes.c_size_t * 7)(*values) if _COPY_SUBMITTER else values

    def __init__(
        self,
        inputs,
        joint,
        pg,
        chunks,
        comm,
        kernel,
        metadata,
        *,
        head_sizes=None,
        joint_strategy="front",
        balanced_peers=False,
    ):
        self.attention_kernel, self.metadata = kernel, metadata
        q = inputs[0]
        batch, local_seq, heads, dim = q.shape
        assert batch == 1 and heads % dist.get_world_size(pg) == 0
        self.inputs, self.joint = inputs, joint
        self.world, self.rank = dist.get_world_size(pg), dist.get_rank(pg)
        self.chunks, self.local_seq, self.dim = chunks, local_seq, dim
        self.joint_len = 0 if joint is None else joint[0].shape[1]
        self.joint_kv_len = 0 if joint is None else joint[1].shape[1]
        if joint_strategy not in ("front", "rear"):
            raise ValueError("The direct plan requires front or rear joint placement")
        self.joint_strategy = joint_strategy
        if joint is not None and not (
            joint[1].shape == joint[2].shape
            and joint[0].shape[:1] + joint[0].shape[2:] == joint[1].shape[:1] + joint[1].shape[2:]
        ):
            raise ValueError("Joint Q/K/V must share batch, heads and dimension; K/V must share length")
        self.sequence_offset = self.joint_len if joint_strategy == "front" else 0
        self.joint_offset = 0 if joint_strategy == "front" else local_seq * self.world
        self.output_joint_offset = 0 if joint_strategy == "front" else local_seq
        if head_sizes is None:
            if heads % (self.world * chunks):
                raise ValueError("Uniform groups must divide the per-rank head count")
            head_sizes = (heads // (self.world * chunks),) * chunks
        self.head_sizes = tuple(head_sizes)
        if (
            len(self.head_sizes) != chunks
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.head_sizes)
            or sum(self.head_sizes) * self.world != heads
        ):
            raise ValueError("Positive group head counts must cover every per-rank head exactly once")
        # Global head blocks include every rank's slice of each group. Build
        # offsets once; forward and reverse must use the same permutation.
        self.head_offsets = tuple(self.world * sum(self.head_sizes[:g]) for g in range(chunks))
        self.stream = comm
        self.runtime = _ce_runtime()
        sequence = local_seq * self.world + max(self.joint_len, self.joint_kv_len)
        output_shape = (1, self.joint_len + local_seq, heads, dim)
        self.forward = [SymmetricBuffer((3, sequence, size, dim), q.dtype, q.device, pg) for size in self.head_sizes]
        self.output = SymmetricBuffer(output_shape, q.dtype, q.device, pg)
        self.ready = [torch.cuda.Event() for _ in range(chunks)]
        self.produced = [torch.cuda.Event() for _ in range(chunks)]
        self.enter, self.done = torch.cuda.Event(), torch.cuda.Event()
        # Cache all fixed QKV transfer geometry and input views outside the call.
        self.forward_ops = [self._forward_geometry(g) for g in range(chunks)]
        self.parts = [
            tuple(
                x.tensor[i, : local_seq * self.world + (self.joint_len if i == 0 else self.joint_kv_len)].unsqueeze(0)
                for i in range(3)
            )
            for x in self.forward
        ]

        self.local_forward, self.remote_forward = [], []
        for buffer, operations in zip(self.forward, self.forward_ops):
            base = buffer.handle.buffer_ptrs[self.rank]
            end = base + buffer.tensor.nbytes
            self.local_forward.append([op for op in operations if base <= op[0] < end])
            self.remote_forward.append([op for op in operations if not base <= op[0] < end])
        self.local_forward_all = [op for group in self.local_forward for op in group]
        # An optional schedule bounds prefetch to the attention currently ready
        # on compute. Realize dependency handles at preparation, before trials.
        self.admitted = [torch.cuda.Event() for _ in range(self.chunks - 1)]
        for event in self.admitted:
            event.record()

        peers = [p for p in range(self.world) if p != self.rank]
        if balanced_peers:
            # Match the native CE order. In each round every rank writes to
            # a different destination. Keep Q, K and V for that peer together.
            order = [
                self.rank ^ step if self.world & (self.world - 1) == 0 else (self.rank + step) % self.world
                for step in range(1, self.world)
            ]
            self.remote_forward = [
                [ops[field * len(peers) + peers.index(peer)] for peer in order for field in range(3)]
                for ops in self.remote_forward
            ]
            peers = order
        if _COPY_SUBMITTER:
            # Descriptor pointers remain valid when bind() updates their source
            # addresses in place. Prepare every forward batch before sampling.
            self.copy_library = ctypes.CDLL(_COPY_SUBMITTER)
            self.submit_copies = self.copy_library.hcms_submit_copies
            self.submit_copies.argtypes = [
                ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            self.submit_copies.restype = ctypes.c_int
            self.memcpy_2d = ctypes.cast(self.runtime.lib.cudaMemcpy2DAsync, ctypes.c_void_p)
            groups = [
                *self.local_forward,
                *self.remote_forward,
                self.local_forward_all,
            ]
            self.copy_batches = {
                tuple(map(id, operations)): (ctypes.POINTER(ctypes.c_size_t) * len(operations))(*operations)
                for operations in groups
            }
        tensors = inputs + (() if joint is None else joint)
        pointers = [x.data_ptr() for x in tensors]
        self.bindings = []
        for operations in self.forward_ops:
            offset = 0
            for field in range(3):
                for operation in operations[offset : offset + self.world]:
                    self.bindings.append((operation, field, operation[2] - pointers[field]))
                offset += self.world
                if joint is not None and joint[field].shape[1]:
                    operation = operations[offset]
                    self.bindings.append((operation, field + 3, operation[2] - pointers[field + 3]))
                    offset += 1
            assert offset == len(operations)

        # Attention outputs change addresses between layers. Keep descriptors
        # and their pointer batches resident; only source addresses/pitches and
        # the selected output slot need rebinding during an actual core call.
        itemsize = inputs[0].element_size()
        self.reverse_ops = []
        for group, width in enumerate(self.head_sizes):
            per_peer = []
            for peer in range(self.world):
                operations = [self.geometry(0, 0, 0, 0, width * self.dim * itemsize, self.local_seq)]
                if self.joint_len:
                    operations.append(self.geometry(0, 0, 0, 0, width * self.dim * itemsize, self.joint_len))
                per_peer.append(operations)
            self.reverse_ops.append(per_peer)
        self.remote_reverse = [[op for peer in peers for op in per_peer[peer]] for per_peer in self.reverse_ops]
        if _COPY_SUBMITTER:
            for per_peer, remote in zip(self.reverse_ops, self.remote_reverse):
                for operations in (per_peer[self.rank], remote):
                    self.copy_batches[tuple(map(id, operations))] = (ctypes.POINTER(ctypes.c_size_t) * len(operations))(
                        *operations
                    )
        self.reverse_rows = [
            ((self.sequence_offset, self.sequence_offset + peer * self.local_seq),)
            + (((self.output_joint_offset, self.joint_offset),) if self.joint_len else ())
            for peer in range(self.world)
        ]
        # Geometry is now pointer-only. The executing core owns its inputs
        # through compute's final transport wait; keeping them in cached plans
        # would retain a different layer's full QKV allocation per candidate.
        # Captured callers separately own their static graph inputs.
        self.inputs, self.joint = (), None

    def _forward_geometry(self, g):
        operations = []
        head_bytes = self.dim * self.inputs[0].element_size()
        width = self.head_sizes[g] * head_bytes
        shared = self.forward[g]
        section_bytes = shared.tensor[0].nbytes
        for index, x in enumerate(self.inputs):
            joint_length = self.joint_len if index == 0 else self.joint_kv_len
            sequence_offset = joint_length if self.joint_strategy == "front" else 0
            source_pitch = x.stride(1) * x.element_size()
            for peer in range(self.world):
                dest = (
                    shared.handle.buffer_ptrs[peer]
                    + index * section_bytes
                    + (sequence_offset + self.rank * self.local_seq) * width
                )
                source = x.data_ptr() + (self.head_offsets[g] + peer * self.head_sizes[g]) * head_bytes
                operations.append(self.geometry(dest, width, source, source_pitch, width, self.local_seq))
            if self.joint is not None and joint_length:
                j = self.joint[index]
                source = j.data_ptr() + (self.head_offsets[g] + self.rank * self.head_sizes[g]) * head_bytes
                dest = shared.handle.buffer_ptrs[self.rank] + index * section_bytes + self.joint_offset * width
                operations.append(
                    self.geometry(
                        dest,
                        width,
                        source,
                        j.stride(1) * j.element_size(),
                        width,
                        joint_length,
                    )
                )
        return operations

    def emit_on(self, args, stream):
        self.runtime.CUDART_CHECK(self.runtime.lib.cudaMemcpy2DAsync(*args, stream.cuda_stream))

    def submit_forward(self, group):
        self.emit_batch_on(self.remote_forward[group], self.stream)
        self.forward[group].handle.barrier()
        self.ready[group].record(self.stream)

    def reverse_operations(self, group, out, peer):
        operations = self.reverse_ops[group][peer]
        itemsize = out.element_size()
        spitch = out.stride(1) * itemsize
        dpitch = self.output.tensor.stride(1) * itemsize
        dest = self.output.handle.buffer_ptrs[peer]
        head_offset = (self.head_offsets[group] + self.rank * self.head_sizes[group]) * self.dim * itemsize
        source = out.data_ptr()
        for operation, (dest_row, source_row) in zip(operations, self.reverse_rows[peer]):
            operation[0] = dest + dest_row * dpitch + head_offset
            operation[1] = dpitch
            operation[2] = source + source_row * spitch
            operation[3] = spitch
        return operations

    def submit_reverse(self, group, out):
        for peer in range(self.world):
            if peer != self.rank:
                self.reverse_operations(group, out, peer)
        self.emit_batch_on(self.remote_reverse[group], self.stream)

    def emit_batch_on(self, operations, stream):
        if not _COPY_SUBMITTER:
            for operation in operations:
                self.emit_on(operation, stream)
            return
        batch = self.copy_batches[tuple(map(id, operations))]
        self.runtime.CUDART_CHECK(self.submit_copies(batch, len(batch), stream.cuda_stream, self.memcpy_2d))

    def attend(self, q, k, v):
        out = self.attention_kernel(q, k, v, self.metadata)
        if out.stride(3) != 1 or out.stride(2) != out.shape[3]:
            out = out.contiguous()
        return out

    def bind(self, inputs, joint, kernel, metadata):
        tensors = inputs + (() if joint is None else joint)
        pointers = [x.data_ptr() for x in tensors]
        for operation, field, offset in self.bindings:
            operation[2] = pointers[field] + offset
        self.attention_kernel, self.metadata = kernel, metadata

    def run(self):
        compute = torch.cuda.current_stream()
        # Split inputs have disjoint local and peer rows; stage local inputs
        # while the first group is in flight. Keep the whole control's order.
        if self.chunks == 1:
            self.emit_batch_on(self.local_forward_all, compute)
        self.enter.record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.enter)
            self.submit_forward(0)
        if self.chunks > 1:
            self.emit_batch_on(self.local_forward_all, compute)
        outputs = []
        for g, parts in enumerate(self.parts):
            compute.wait_event(self.ready[g])
            out = self.attend(*parts)
            outputs.append(out)
            self.produced[g].record(compute)
            with torch.cuda.stream(self.stream):
                if g + 1 < self.chunks:
                    self.submit_forward(g + 1)
                if g:
                    self.stream.wait_event(self.produced[g - 1])
                    self.submit_reverse(g - 1, outputs[g - 1])
            self.emit_batch_on(self.reverse_operations(g, out, self.rank), compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.produced[-1])
            self.submit_reverse(self.chunks - 1, outputs[-1])
            self.output.handle.barrier()
            self.done.record(self.stream)
        compute.wait_event(self.done)
        return self.output.tensor


def prepare_plan(inputs, joint, pg, head_sizes, comm, kernel, metadata, *, joint_strategy="front", query_tail_rows=0):
    """Allocate each candidate once at a component boundary, before any timed layer."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Cannot prepare an adaptive head plan during CUDA graph capture")
    options = {}
    pipeline = HeadPlan
    if query_tail_rows:
        from .query_tail import QueryTailPipeline

        pipeline = QueryTailPipeline
        options["tail_rows"] = query_tail_rows
    return pipeline(
        inputs,
        joint,
        pg,
        len(head_sizes),
        comm,
        kernel,
        metadata,
        head_sizes=head_sizes,
        joint_strategy=joint_strategy,
        balanced_peers=True,
        **options,
    )
