# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager NPU Ulysses using Triton and the installed symmetric-memory runtime.

No repository C++/AscendC extension is compiled or loaded. Requires Triton-Ascend,
TorchNPU's NPUSHMEM allocator and libshmem.so on the loader search path. All
WORLD ranks must issue the same exchanges on one stream in one HCCS domain.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
_WORKSPACE = None
_WORKSPACE_GROUP = None


@cache
def _runtime_library() -> ctypes.CDLL:
    try:
        library = ctypes.CDLL("libshmem.so")
    except OSError as exc:
        raise RuntimeError("Install the symmetric-memory runtime and put libshmem.so on LD_LIBRARY_PATH") from exc
    barrier = library.aclshmemx_barrier_all_on_stream
    barrier.argtypes = [ctypes.c_void_p]
    barrier.restype = None
    return library


def ensure_a2a_permute_available() -> None:
    """Check dependencies without building a repository-owned extension."""
    _runtime_library()
    if triton.runtime.driver.active.get_current_target().backend not in ("npu", "ascend"):
        raise RuntimeError("NPU symmetric-memory Ulysses requires Triton-Ascend")


@triton.jit
def _pull_tile(
    peers,
    output,
    rank,
    world: tl.constexpr,
    seq: tl.constexpr,
    channels: tl.constexpr,
    reverse: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    batch,
    peer,
    tile,
    offset: tl.constexpr,
    interleave: tl.constexpr = 1,
):
    s = tile * rows + tl.arange(0, rows)[:, None]
    c = tl.arange(0, columns)[None, :]
    address = tl.load(peers + peer)
    if reverse:
        base = offset + (batch * world + rank) * seq * channels
        src = s * channels + c
        dst = ((batch * seq + s) * world + peer) * channels + c
    else:
        base = offset + (batch * seq * world * interleave + rank) * channels
        src = s * world * interleave * channels + c
        dst = ((batch * world + peer) * seq + s) * channels + c
    # Form the row base in integer address space before creating the pointer.
    # This also handles 128-byte GQA rank offsets that the Ascend compiler
    # otherwise folds into an incompatible nonzero memref offset.
    element_bytes: tl.constexpr = 4 if output.dtype.element_ty == tl.float32 else 2
    source = (address + base.to(tl.uint64) * element_bytes).to(tl.pointer_type(output.dtype.element_ty))
    mask = (s < seq) & (c < channels)
    values = tl.load(source + src, mask, other=0)
    tl.store(output + dst, values, mask)


@triton.jit
def _pull(
    peers,
    q,
    k,
    v,
    rank,
    world: tl.constexpr,
    batch: tl.constexpr,
    seq: tl.constexpr,
    qc: tl.constexpr,
    kc: tl.constexpr,
    vc: tl.constexpr,
    q_columns: tl.constexpr,
    k_columns: tl.constexpr,
    v_columns: tl.constexpr,
    reverse: tl.constexpr,
    count: tl.constexpr,
    rows: tl.constexpr,
    packed: tl.constexpr = False,
):
    tiles: tl.constexpr = triton.cdiv(seq, rows)
    # Tile-major order spreads adjacent jobs across peers. A persistent grid
    # bounds concurrent remote readers instead of launching one core per tile.
    for job in range(tl.program_id(0), batch * world * tiles * count, tl.num_programs(0)):
        peer = job % world
        tile = (job // world) % tiles
        b = (job // (world * tiles)) % batch
        item = job // (world * tiles * batch)
        if count == 1:
            _pull_tile(peers, q, rank, world, seq, qc, reverse, rows, q_columns, b, peer, tile, 0)
        elif item == 0:
            _pull_tile(
                peers,
                q,
                rank,
                world,
                seq,
                qc,
                False,
                rows,
                q_columns,
                b,
                peer,
                tile,
                0,
                3 if packed else 1,
            )
        elif item == 1:
            _pull_tile(
                peers,
                k,
                rank,
                world,
                seq,
                kc,
                False,
                rows,
                k_columns,
                b,
                peer,
                tile,
                (1 if packed else batch * seq) * world * qc,
                3 if packed else 1,
            )
        else:
            _pull_tile(
                peers,
                v,
                rank,
                world,
                seq,
                vc,
                False,
                rows,
                v_columns,
                b,
                peer,
                tile,
                (1 if packed else batch * seq) * world * (qc + kc),
                3 if packed else 1,
            )


@dataclass(slots=True)
class _ExchangePlan:
    dtype: torch.dtype
    shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    packed: bool
    staging: tuple[torch.Tensor, ...]
    launch: Callable[..., None]


class NPUUlyssesSymmetricMemory:
    """Owned outputs over one eager WORLD workspace.

    Each exchange fences publication before peer reads and fences reuse after
    them. One symmetric staging buffer is shared across layers and shapes.
    """

    def __init__(self, capacity_bytes: int) -> None:
        import torch_npu

        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise RuntimeError("Symmetric memory exchange requires an initialized WORLD with at least two ranks")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        ensure_a2a_permute_available()
        self._npu = torch_npu._C
        if self._npu._npu_isCurrentStreamCapturing():
            raise RuntimeError("NPU symmetric memory Ulysses requires eager execution; graph capture is unsupported")
        self.device = torch.device("npu", self._npu._npu_getDevice())
        if symm_mem.get_backend(self.device) != "NPUSHMEM":
            raise RuntimeError("TorchNPU with the NPUSHMEM symmetric-memory allocator is required")
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.capacity_bytes = capacity_bytes
        self.stream = torch.npu.current_stream(self.device)
        # Use the same existing bindings as current_stream(), without rebuilding
        # its Python Stream wrapper and resolving a generic accelerator device.
        self._stream_key = self._npu._npu_getCurrentStream(self.device.index)
        self._barrier = _runtime_library().aclshmemx_barrier_all_on_stream
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        self.allocation = symm_mem.empty(capacity_bytes, dtype=torch.uint8, device=self.device)
        self.handle = symm_mem.rendezvous(self.allocation, dist.group.WORLD.group_name)
        pointers = self.handle.buffer_ptrs
        if len(pointers) != self.world or any(ptr == 0 for ptr in pointers):
            raise RuntimeError("All WORLD peers must be directly accessible in one HCCS domain")
        self.peer_ptrs = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        self._raw_stream = int(self.stream.npu_stream)
        self._plans: dict[tuple[bool, int], _ExchangePlan] = {}
        self._closed = False

    def _check_stream(self) -> None:
        if self._closed:
            raise RuntimeError("Symmetric memory workspace is closed")
        if self._npu._npu_getDevice() != self.device.index:
            raise RuntimeError("Symmetric memory workspace must be used on its original NPU device")
        if self._npu._npu_getCurrentStream(self.device.index) != self._stream_key:
            raise RuntimeError("Symmetric memory workspace must be used on a single NPU stream")
        if self._npu._npu_isCurrentStreamCapturing():
            raise RuntimeError("NPU symmetric memory Ulysses requires eager execution; graph capture is unsupported")

    def _fence(self) -> None:
        # This fresh property lookup drains TorchNPU's host task queue before
        # direct ACL submission. Caching this value here would break ordering.
        self._barrier(ctypes.c_void_p(int(self.stream.npu_stream)))

    def _row_bytes(self, tensor: torch.Tensor, reverse: bool = False) -> int:
        if tensor.device != self.device:
            raise ValueError("Input must be on the workspace NPU device")
        if tensor.requires_grad:
            raise ValueError("NPU symmetric memory Ulysses is inference-only")
        if tensor.ndim != 4 or any(tensor.shape[i] <= 0 for i in (0, 2, 3)):
            raise ValueError("Input must be [B, S, H, D] with positive batch, heads and head size")
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Supported dtypes: bfloat16, float16, float32")
        if reverse:
            if tensor.shape[1] % self.world:
                raise ValueError("Global sequence length must be divisible by WORLD size")
        elif tensor.shape[2] % self.world:
            raise ValueError("Head count must be divisible by WORLD size; pad/repeat GQA heads upstream")
        row_bytes = tensor.shape[2] * tensor.shape[3] * tensor.element_size() // (1 if reverse else self.world)
        if row_bytes % 32 or row_bytes > 32768:
            raise ValueError("Head shard rows must be 32-byte aligned and at most 32 KiB")
        return row_bytes

    @staticmethod
    def _packed_input(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
        if len(tensors) != 3 or tensors[0].shape != tensors[1].shape or tensors[0].shape != tensors[2].shape:
            return None
        query, key, value = tensors
        batch, seq, heads, dim = query.shape
        row, stride = heads * dim, query.stride()
        if stride[1:] != (3 * row, dim, 1) or key.stride() != stride or value.stride() != stride:
            return None
        pointer, step = query.data_ptr(), row * query.element_size()
        if key.data_ptr() != pointer + step or value.data_ptr() != pointer + 2 * step:
            return None
        return query.as_strided((batch, seq, 3, heads, dim), (stride[0], stride[1], row, dim, 1))

    def _make_plan(
        self,
        tensors: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
        reverse: bool,
        packed: bool,
    ) -> _ExchangePlan:
        query = tensors[0]
        batch, seq = query.shape[:2]
        seq_local = seq // self.world if reverse else seq
        sizes = [tensor.numel() for tensor in tensors]
        required = sum(sizes) * query.element_size()
        if required > self.capacity_bytes:
            raise ValueError(f"Input requires {required} bytes; capacity is {self.capacity_bytes}")
        flat = self.allocation[:required].view(query.dtype)
        if packed:
            staging = (flat.view(batch, seq, 3, *query.shape[2:]),)
        else:
            views = []
            offset = 0
            for tensor, size in zip(tensors, sizes):
                views.append(flat[offset : offset + size].view(tensor.shape))
                offset += size
            staging = tuple(views)
        channels = [tensor.shape[2] * tensor.shape[3] // (1 if reverse else self.world) for tensor in tensors]
        qc, kc, vc = (channels * 3)[:3] if len(channels) == 1 else channels
        # Resolve Python helpers here: some Ascend JIT contexts pass constexpr
        # wrappers to next_power_of_2(), which cannot evaluate bit_length().
        q_columns, k_columns, v_columns = (triton.next_power_of_2(c) for c in (qc, kc, vc))
        rows = max(1, min(32, 32768 // (max(q_columns, k_columns, v_columns) * query.element_size())))
        rows = 2 ** (rows.bit_length() - 1)
        # Bound reader concurrency on large transfers; small transfers benefit
        # from more readers. This is independent of a particular SoC or rank count.
        workers = (4 if required <= (16 << 20) else 2) * self.world
        q, k, v = (outputs * 3)[:3] if len(outputs) == 1 else outputs
        kernel = _pull.warmup(
            self.peer_ptrs,
            q,
            k,
            v,
            self.rank,
            self.world,
            batch,
            seq_local,
            qc,
            kc,
            vc,
            q_columns,
            k_columns,
            v_columns,
            reverse,
            len(tensors),
            rows,
            packed,
            grid=(workers,),
        )
        return _ExchangePlan(
            query.dtype,
            tuple(tuple(t.shape) for t in tensors),
            tuple(tuple(o.shape) for o in outputs),
            packed,
            staging,
            kernel[(workers, 1, 1)],
        )

    def _exchange(self, tensors: tuple[torch.Tensor, ...], reverse: bool) -> tuple[torch.Tensor, ...]:
        self._check_stream()
        query = tensors[0]
        for tensor in tensors:
            if tensor.device != self.device or tensor.requires_grad:
                raise ValueError("Inputs must be inference tensors on the workspace NPU device")
            if tensor.dtype != query.dtype:
                raise ValueError("Q/K/V must share dtype")
        shapes = tuple(tuple(tensor.shape) for tensor in tensors)
        kind = (reverse, len(tensors))
        plan = self._plans.get(kind)
        if plan is None or plan.dtype != query.dtype or plan.shapes != shapes:
            for tensor in tensors:
                self._row_bytes(tensor, reverse)
                if tensor.shape[:2] != query.shape[:2]:
                    raise ValueError("Q/K/V must share batch/sequence dimensions")
            required = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            if required > self.capacity_bytes:
                raise ValueError(f"Input requires {required} bytes; capacity is {self.capacity_bytes}")
            batch, seq = query.shape[:2]
            seq_local = seq // self.world if reverse else seq
            output_shapes: tuple[tuple[int, ...], ...] = tuple(
                (batch, seq_local, t.shape[2] * self.world, t.shape[3])
                if reverse
                else (batch, seq * self.world, t.shape[2] // self.world, t.shape[3])
                for t in tensors
            )
        else:
            output_shapes = plan.output_shapes
        outputs = tuple(tensor.new_empty(shape) for tensor, shape in zip(tensors, output_shapes))
        if query.shape[1] == 0:
            return outputs
        packed_source = self._packed_input(tensors) if not reverse else None
        packed = packed_source is not None
        if plan is None or plan.dtype != query.dtype or plan.shapes != shapes or plan.packed != packed:
            plan = self._make_plan(tensors, outputs, reverse, packed)
            self._plans[kind] = plan
        if packed_source is not None:
            plan.staging[0].copy_(packed_source)
        else:
            for target, source in zip(plan.staging, tensors):
                target.copy_(source)
        self._fence()
        q, k, v = (outputs * 3)[:3] if len(outputs) == 1 else outputs
        plan.launch(self.peer_ptrs, q, k, v, self.rank, stream=self._raw_stream)
        self._fence()
        return outputs

    def __call__(self, tensor: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
        return self._exchange((tensor,), reverse)[0]

    def qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self._exchange((query, key, value), False)

    def close(self) -> None:
        """Collectively complete peer reads before releasing staging memory."""
        self._check_stream()
        self._fence()
        torch.npu.synchronize(self.device)
        self._plans.clear()
        self.peer_ptrs = None
        self.handle = None
        self.allocation = None
        self._closed = True


def _get_workspace(tensors: tuple[torch.Tensor, ...], group_name: str, world_size: int) -> NPUUlyssesSymmetricMemory:
    """Share one workspace across layers; grow collectively for larger requests.

    Strict Ulysses already requires matching tensor shapes across ranks. The
    resulting byte capacity and growth decisions must consequently match too.
    The group must contain every WORLD rank in the same order, since the
    device barrier is collective over WORLD.
    """
    global _WORKSPACE, _WORKSPACE_GROUP
    group = _resolve_group(group_name, world_size)
    if tensors[0].device.type != "npu":
        raise ValueError("NPU ulysses_a2a_permute requires NPU tensors")
    if _WORKSPACE_GROUP is not None and group is not _WORKSPACE_GROUP:
        raise RuntimeError(
            "NPU symmetric memory Ulysses supports a single process group; clear it before reinitialization"
        )
    if _WORKSPACE is None:
        if dist.get_process_group_ranks(group) != list(range(dist.get_world_size())):
            raise ValueError("NPU symmetric memory Ulysses requires the Ulysses group to match WORLD rank order")
    required = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if _WORKSPACE is not None:
        if required <= _WORKSPACE.capacity_bytes:
            return _WORKSPACE
        _WORKSPACE.close()
        _WORKSPACE = None
    # The first exchange may be a joint-only CFG prefill with zero image
    # tokens. Rendezvous still needs a positive symmetric allocation.
    capacity = max(64 << 20, ((required + (64 << 20) - 1) // (64 << 20)) * (64 << 20))
    _WORKSPACE = NPUUlyssesSymmetricMemory(capacity)
    _WORKSPACE_GROUP = group
    logger.info(
        "NPU symmetric memory Ulysses initialized (rank=%d, world=%d, capacity_bytes=%d)",
        dist.get_rank(),
        dist.get_world_size(),
        capacity,
    )
    return _WORKSPACE


def _resolve_group(group_name: str, world_size: int):
    group = _resolve_process_group(group_name)
    if dist.get_world_size(group) != world_size:
        raise ValueError("world_size must match the Ulysses process group")
    return group


def clear_ulysses_symmetric_memory_workspace() -> None:
    """Collectively release the workspace before destroying process groups."""
    global _WORKSPACE, _WORKSPACE_GROUP
    if _WORKSPACE is not None:
        _WORKSPACE.close()
        _WORKSPACE = None
    _WORKSPACE_GROUP = None


def ulysses_qkv_fwd(x: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return _get_workspace((x,), group_name, world_size)(x)


def ulysses_qkv_fwd_group(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group_name: str,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _get_workspace((query, key, value), group_name, world_size).qkv(query, key, value)


def ulysses_o_rev(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return _get_workspace((y,), group_name, world_size)(y, reverse=True)
