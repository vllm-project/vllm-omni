# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Ulysses all-to-all over symmetric memory.

Uses CUDA copy engines, exposed through functional custom ops:

    ulysses_qkv_ce(q, k, v, group_name, world_size)  # grouped CE forward
    ulysses_o_rev(y, group_name, world_size)     # (B, S, H/p, D) -> (B, S/p, H, D)

These replace the synchronous ``all_to_all_4D`` (permute + NCCL all_to_all_single)
used by Ulysses SP. All symmetric-memory bookkeeping (a grow-only workspace
and transport) lives *inside* the ops, so from torch.compile's point
of view each op is an opaque ``input -> output`` function: no graph break, no
visible buffer mutation, no Python control flow in the traced graph. A
``register_fake`` provides output metadata so Dynamo keeps the op in-graph.
"""

from __future__ import annotations

import ctypes
import math
import threading
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

logger = init_logger(__name__)

_WORKSPACE_LOCK = threading.Lock()
# Bound graph staging per group to 15 MiB for equal-shape Q/K/V and 4 MiB
# for O (fixed inputs only, excluding allocator rounding).
_CE_GRAPH_MAX_INPUT_BYTES = 4 * 1024 * 1024
_CE_QKV_GRAPH_MAX_INPUT_BYTES = 5 * 1024 * 1024


@dataclass(slots=True)
class _SymmWorkspace:
    allocation: torch.Tensor
    handle: Any
    capacity_bytes: int
    stream_id: int
    views: dict[tuple[tuple[int, ...], torch.dtype], torch.Tensor] = field(default_factory=dict)
    ce_plans: dict[bool, _CEPlan] = field(default_factory=dict)
    ce_stream: torch.cuda.Stream | None = None


# A single grow-only byte workspace per device/process-group. Shape changes
# reuse a typed view into the workspace instead of retaining one allocation per
# historical request shape.
_SYMM_WORKSPACES: dict[tuple[torch.device, str], _SymmWorkspace] = {}


class _Extent3D(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in ("width", "height", "depth")]


class _PitchedPtr(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p)] + [(name, ctypes.c_size_t) for name in ("pitch", "xsize", "ysize")]


class _Memcpy3DParams(ctypes.Structure):
    _fields_ = [
        ("srcArray", ctypes.c_void_p),
        ("srcPos", _Extent3D),
        ("srcPtr", _PitchedPtr),
        ("dstArray", ctypes.c_void_p),
        ("dstPos", _Extent3D),
        ("dstPtr", _PitchedPtr),
        ("extent", _Extent3D),
        ("kind", ctypes.c_int),
    ]


@cache
def _ce_runtime() -> CudaRTLibrary:
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_cuda():
        raise ValueError("CE Ulysses requires the CUDA platform")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    if symm_mem.get_backend(device) != "CUDA":
        symm_mem.set_backend("CUDA")
    runtime = CudaRTLibrary()
    copy = runtime.lib.cudaMemcpy2DAsync
    copy.restype = ctypes.c_int
    copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    copy3d = runtime.lib.cudaMemcpy3DAsync
    copy3d.restype = ctypes.c_int
    copy3d.argtypes = [ctypes.POINTER(_Memcpy3DParams), ctypes.c_void_p]
    return runtime


def ensure_a2a_permute_available() -> None:
    """Initialize the CUDA copy runtime before serving requests."""
    _ce_runtime()


def _required_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * dtype.itemsize


def _get_symm_buffer(
    symm_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, group_name: str
) -> torch.Tensor:
    """Get a typed view from a grow-only symmetric-memory workspace.

    Allocation/growth and rendezvous are collective. Every rank reaches them in
    lockstep because sequence parallel execution is symmetric. Steady-state
    requests at or below the peak capacity take no synchronization path.
    """
    device = torch.device(device)
    key = (device, group_name)
    required_bytes = _required_nbytes(symm_shape, dtype)
    stream_id = torch.cuda.current_stream(device).cuda_stream
    with _WORKSPACE_LOCK:
        workspace = _SYMM_WORKSPACES.get(key)
        if workspace is not None and workspace.stream_id != stream_id:
            raise RuntimeError("a2a_permute workspace must be used from a single CUDA stream")
        if workspace is None or workspace.capacity_bytes < required_bytes:
            if torch.cuda.is_current_stream_capturing():
                capacity = 0 if workspace is None else workspace.capacity_bytes
                raise RuntimeError(
                    "a2a_permute: cannot grow the symmetric-memory workspace "
                    f"from {capacity} to {required_bytes} bytes during CUDA graph capture; "
                    "warm up the maximum request shape before capture"
                )
            if workspace is None:
                # Initialize the process-group communicator before rendezvous.
                pg = _resolve_process_group(group_name)
                warm = torch.ones(1, device=device)
                dist.all_reduce(warm, group=pg)
            # Growth is rare and must not release the previous allocation while
            # an earlier kernel is still consuming it.
            torch.accelerator.synchronize(device)
            # Peer pointers must identify this allocation, not the base of
            # an implicit pool containing it at a nonzero offset.
            allocation = symm_mem._SymmetricMemory.empty_strided_p2p((required_bytes,), (1,), torch.uint8, device)
            handle = symm_mem.rendezvous(allocation, group_name)
            workspace = _SymmWorkspace(
                allocation=allocation,
                handle=handle,
                capacity_bytes=required_bytes,
                stream_id=stream_id,
            )
            _SYMM_WORKSPACES[key] = workspace
        view_key = (symm_shape, dtype)
        buffer = workspace.views.get(view_key)
        if buffer is not None:
            return buffer
        buffer = workspace.handle.get_buffer(workspace.handle.rank, symm_shape, dtype)
        # Peer writes target the allocation base; keep the local view aligned
        # even when the handle exposes a window-relative pointer.
        if isinstance(buffer, torch.Tensor) and buffer.data_ptr() != workspace.allocation.data_ptr():
            buffer = workspace.allocation[:required_bytes].view(dtype).view(symm_shape)
        # Keep at most the forward and reverse views; capacity stays grow-only
        # without retaining a view for every historical request shape.
        if len(workspace.views) == 2:
            workspace.views.clear()
        workspace.views[view_key] = buffer
        return buffer


def clear_a2a_permute_workspaces() -> None:
    """Release cached symmetric-memory workspaces during worker shutdown."""
    with _WORKSPACE_LOCK:
        devices = {workspace.allocation.device for workspace in _SYMM_WORKSPACES.values()}
        for device in devices:
            torch.accelerator.synchronize(device)
        count = len(_SYMM_WORKSPACES)
        _SYMM_WORKSPACES.clear()
    logger.info("[a2a_permute] Released %d symmetric-memory workspace(s)", count)


class _CEPlan:
    """Cache copy geometry and stream resources for the latest layout per direction."""

    def __init__(self, inputs, shared, handle, world_size, reverse, transfer):
        self.signature = tuple((x.shape, x.stride(), x.dtype) for x in inputs)
        self.shared = shared
        self.handle = handle
        self.current = torch.cuda.current_stream(shared.device)
        self.transfer = transfer
        self.ready, self.done = torch.cuda.Event(), torch.cuda.Event()
        self.runtime = _ce_runtime()
        self.reverse = reverse
        self.graph: torch.cuda.CUDAGraph | None = None
        self.coalesce_inputs = len(inputs) == 3 and inputs[0].shape == inputs[1].shape == inputs[2].shape
        limit = _CE_QKV_GRAPH_MAX_INPUT_BYTES if self.coalesce_inputs else _CE_GRAPH_MAX_INPUT_BYTES
        self.use_graph = max(x.nbytes for x in inputs) <= limit
        p, rank = world_size, handle.rank
        order = [rank ^ i for i in range(p)] if p & (p - 1) == 0 else [(rank + i) % p for i in range(p)]
        self.remote = order[1:]
        self.operations: list[list[tuple[int, int, list[int] | _Memcpy3DParams]]] = [[] for _ in range(p)]
        section = 0
        for index, x in enumerate(inputs):
            batch, sequence, heads, dim = x.shape
            width = (heads if reverse else heads // p) * dim * x.element_size()
            rows = sequence // p if reverse else sequence
            source_pitch = x.stride(1) * x.element_size()
            dest_pitch = p * width if reverse else width
            for peer in range(p):
                dest = handle.buffer_ptrs[peer] + section + rank * (width if reverse else rows * width)
                offset = peer * (rows * source_pitch if reverse else width)
                args: list[int] | _Memcpy3DParams
                if batch == 1:
                    args = [dest, dest_pitch, 0, source_pitch, width, rows, 4]
                else:
                    args = _Memcpy3DParams()
                    args.srcPtr = _PitchedPtr(None, source_pitch, source_pitch, sequence)
                    args.dstPtr = _PitchedPtr(dest, dest_pitch, dest_pitch, rows if reverse else p * rows)
                    args.extent = _Extent3D(width, rows, batch)
                    args.kind = 4  # cudaMemcpyDefault, using symmetric peer mappings.
                self.operations[peer].append((index, offset, args))
            section += x.nbytes

    def run(self, inputs):
        # An outer graph already amortizes submission costs. Avoid nested
        # capture and staging in that case; large requests use direct CE.
        if not self.use_graph or torch.cuda.is_current_stream_capturing():
            return self._run_eager(inputs)
        if self.graph is None:
            if self.coalesce_inputs:
                batch, sequence, heads, dim = inputs[0].shape
                self.staging = inputs[0].new_empty(3 * batch, sequence, heads, dim)
                self.static_inputs = self.staging.view(3, batch, sequence, heads, dim).unbind(0)
            else:
                self.static_inputs = tuple(torch.empty_like(x, memory_format=torch.contiguous_format) for x in inputs)
            capture_stream = torch.cuda.Stream(device=self.shared.device)
            capture_stream.wait_stream(self.current)
            with torch.cuda.stream(capture_stream):
                self.graph_plan = _CEPlan(
                    self.static_inputs, self.shared, self.handle, len(self.operations), self.reverse, self.transfer
                )
                self.graph_plan._run_eager(self.static_inputs, copy_out=False)
            capture_stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=capture_stream):
                self.graph_plan._run_eager(self.static_inputs, copy_out=False)
            self.current.wait_stream(capture_stream)
        if self.coalesce_inputs:
            torch.cat(inputs, dim=0, out=self.staging)
        else:
            for fixed, incoming in zip(self.static_inputs, inputs):
                fixed.copy_(incoming)
        self.graph.replay()
        # The next exchange's initial barrier protects this copy-out. Clone
        # directly from symmetric memory to avoid retaining a graph output.
        return self.shared.clone()

    def _run_eager(self, inputs, *, copy_out=True):
        pointers = [x.data_ptr() for x in inputs]
        out = torch.empty_like(self.shared) if copy_out else None
        runtime = self.runtime

        def emit(peer, stream):
            for index, offset, args in self.operations[peer]:
                source = pointers[index] + offset
                if isinstance(args, list):
                    args[2] = source
                    result = runtime.lib.cudaMemcpy2DAsync(*args, stream)
                else:
                    args.srcPtr.ptr = source
                    result = runtime.lib.cudaMemcpy3DAsync(ctypes.byref(args), stream)
                runtime.CUDART_CHECK(result)

        # Peers must finish their previous copy-outs before we overwrite them.
        self.handle.barrier()
        self.ready.record(self.current)
        self.transfer.wait_event(self.ready)
        for peer in self.remote:
            emit(peer, self.transfer.cuda_stream)
        emit(self.handle.rank, self.current.cuda_stream)
        self.done.record(self.transfer)
        self.current.wait_event(self.done)
        self.handle.barrier()
        # Ordinary outputs survive later exchanges and satisfy custom-op aliasing.
        if out is not None:
            out.copy_(self.shared)
        return out


def _exchange_ce(inputs, group_name: str, world_size: int, reverse: bool = False) -> torch.Tensor:
    # Reuse a validated plan before repeating layout and workspace bookkeeping.
    workspace = _SYMM_WORKSPACES.get((inputs[0].device, group_name))
    if workspace is not None:
        plan = workspace.ce_plans.get(reverse)
        signature = tuple((x.shape, x.stride(), x.dtype) for x in inputs)
        if plan is not None and plan.signature == signature:
            stream_id = torch.cuda.current_stream(inputs[0].device).cuda_stream
            if workspace.stream_id != stream_id:
                raise RuntimeError("a2a_permute workspace must be used from a single CUDA stream")
            return plan.run(inputs)
    _ce_runtime()
    # Preserve regular row-strided QKV views; pack only unsupported layouts.
    inputs = tuple(
        x
        if (
            x.stride(3) == 1 and x.stride(2) == x.size(3) and (x.size(0) == 1 or x.stride(0) == x.size(1) * x.stride(1))
        )
        else x.contiguous()
        for x in inputs
    )
    first = inputs[0]
    shared = _get_symm_buffer((sum(x.numel() for x in inputs),), first.dtype, first.device, group_name)
    workspace = _SYMM_WORKSPACES[(first.device, group_name)]
    if workspace.ce_stream is None:
        workspace.ce_stream = torch.cuda.Stream(device=first.device)
    signature = tuple((x.shape, x.stride(), x.dtype) for x in inputs)
    plan = workspace.ce_plans.get(reverse)
    if plan is None or plan.signature != signature:
        plan = _CEPlan(inputs, shared, workspace.handle, world_size, reverse, workspace.ce_stream)
        workspace.ce_plans[reverse] = plan
    return plan.run(inputs)


@torch.library.custom_op("vllm_omni_a2a::ulysses_qkv_ce", mutates_args=(), device_types="cuda")
def ulysses_qkv_ce(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, group_name: str, world_size: int
) -> torch.Tensor:
    """Push Q/K/V together into packed, batch-major symmetric outputs."""
    return _exchange_ce((query, key, value), group_name, world_size)


@ulysses_qkv_ce.register_fake
def _(query, key, value, group_name, world_size):
    return query.new_empty(query.numel() + key.numel() + value.numel())


# ---------------------------------------------------------------------------
# Reverse: (B, S_global, H_local, D) -> (B, S_local, H, D)
#   scatter sequence (dim 1), gather heads (dim 2). Matches all_to_all_4D(y,1,2).
# ---------------------------------------------------------------------------
@torch.library.custom_op("vllm_omni_a2a::ulysses_o_rev", mutates_args=(), device_types="cuda")
def ulysses_o_rev(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    B, S, H, D = y.shape
    return _exchange_ce((y,), group_name, world_size, reverse=True).view(B, S // world_size, H * world_size, D)


@ulysses_o_rev.register_fake
def _(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    B, s_global, Hl, D = y.shape
    return y.new_empty(B, s_global // world_size, Hl * world_size, D)
