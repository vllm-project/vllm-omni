# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager NPU Ulysses transport using SHMEM remote reads fused with relayout.

Loaded only when ``ulysses_a2a_permute`` is enabled on NPU. Requires TorchNPU's NPUSHMEM
allocator, libshmem.so on LD_LIBRARY_PATH, Triton-Ascend and a C++ toolchain for
the queue-aware stream barrier launcher. All WORLD ranks must participate in
the same order on one stream in one HCCS domain. Independent SP groups, graph
capture and autograd are unsupported. Head-shard rows must be 32-byte aligned.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

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
def _load_barrier() -> None:
    import torch_npu
    from torch.utils.cpp_extension import load

    npu_root = Path(torch_npu.__file__).parent
    cann_root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    load(
        name="omni_ulysses_shmem_barrier",
        sources=[str(Path(__file__).parent / "csrc/ulysses_shmem_barrier.cpp")],
        extra_include_paths=[
            str(npu_root / "include"),
            str(npu_root / "include/third_party/acl/inc"),
            str(cann_root / "include"),
        ],
        extra_ldflags=[f"-L{npu_root / 'lib'}", "-ltorch_npu", "-ldl"],
        extra_cflags=["-O2"],
        with_cuda=False,
        is_python_module=False,
        verbose=False,
    )


def ensure_a2a_permute_available() -> None:
    """Prepare the NPU extension without building the CUDA implementation."""
    _load_barrier()


@triton.jit
def _pull_tile(
    peer_ptrs,
    output,
    rank: tl.constexpr,
    world: tl.constexpr,
    seq_local: tl.constexpr,
    channels: tl.constexpr,
    reverse: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
    batch,
    source_offset: tl.constexpr,
):
    peer = tl.program_id(1)
    row_index = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    row = row_index[:, None]
    column = tl.arange(0, block_channels)[None, :]
    source = tl.load(peer_ptrs + peer).to(tl.pointer_type(output.dtype.element_ty))
    source += source_offset
    if reverse:
        # Each peer owns all sequences and its own head shard.
        src_offset = ((batch * world + rank) * seq_local + row) * channels + column
        dst_offset = ((batch * seq_local + row) * world + peer) * channels + column
    else:
        # Each peer owns its sequence shard and all heads.
        src_offset = ((batch * seq_local + row) * world + rank) * channels + column
        dst_offset = ((batch * world + peer) * seq_local + row) * channels + column
    mask = (row_index[:, None] < seq_local) & (column < channels)
    value = tl.load(source + src_offset, mask, other=0)
    tl.store(output + dst_offset, value, mask)


@triton.jit
def _pull_and_relayout(
    peer_ptrs,
    output,
    rank: tl.constexpr,
    world: tl.constexpr,
    seq_local: tl.constexpr,
    channels: tl.constexpr,
    reverse: tl.constexpr,
    block_rows: tl.constexpr,
    block_channels: tl.constexpr,
):
    _pull_tile(
        peer_ptrs,
        output,
        rank,
        world,
        seq_local,
        channels,
        reverse,
        block_rows,
        block_channels,
        tl.program_id(2),
        0,
    )


@triton.jit
def _pull_qkv_and_relayout(
    peer_ptrs,
    query_out,
    key_out,
    value_out,
    rank: tl.constexpr,
    world: tl.constexpr,
    seq_local: tl.constexpr,
    query_channels: tl.constexpr,
    key_channels: tl.constexpr,
    value_channels: tl.constexpr,
    key_offset: tl.constexpr,
    value_offset: tl.constexpr,
    block_rows: tl.constexpr,
    query_block_channels: tl.constexpr,
    key_block_channels: tl.constexpr,
    value_block_channels: tl.constexpr,
):
    batch = tl.program_id(2) // 3
    item = tl.program_id(2) % 3
    if item == 0:
        _pull_tile(
            peer_ptrs,
            query_out,
            rank,
            world,
            seq_local,
            query_channels,
            False,
            block_rows,
            query_block_channels,
            batch,
            0,
        )
    elif item == 1:
        _pull_tile(
            peer_ptrs,
            key_out,
            rank,
            world,
            seq_local,
            key_channels,
            False,
            block_rows,
            key_block_channels,
            batch,
            key_offset,
        )
    else:
        _pull_tile(
            peer_ptrs,
            value_out,
            rank,
            world,
            seq_local,
            value_channels,
            False,
            block_rows,
            value_block_channels,
            batch,
            value_offset,
        )


class NPUUlyssesSHMEM:
    """A fixed-capacity, single-stream workspace with out-of-place outputs.

    Construction and close are collective over WORLD. The capacity must be
    identical on every rank; dtype and shapes may vary below that capacity.
    The two stream barriers publish staging writes and protect reuse after all
    peers have finished reading. A CPU barrier alone would not provide this
    ordering for queued kernels.
    """

    def __init__(self, capacity_bytes: int, *, block_size: int = 4096) -> None:
        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise RuntimeError("SHMEM exchange requires an initialized WORLD with at least two ranks")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("NPU SHMEM Ulysses requires eager execution; graph capture is unsupported")
        self.device = torch.device("npu", torch.npu.current_device())
        if symm_mem.get_backend(self.device) != "NPUSHMEM":
            raise RuntimeError("TorchNPU with the NPUSHMEM symmetric-memory allocator is required")
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        self.capacity_bytes = capacity_bytes
        self.block_size = block_size
        self.stream = torch.npu.current_stream(self.device)
        _load_barrier()
        self._barrier = torch.ops.omni_ulysses_shmem.barrier
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        self.allocation = symm_mem.empty(capacity_bytes, dtype=torch.uint8, device=self.device)
        self.handle = symm_mem.rendezvous(self.allocation, dist.group.WORLD.group_name)
        if len(self.handle.buffer_ptrs) != self.world or any(ptr == 0 for ptr in self.handle.buffer_ptrs):
            raise RuntimeError("All WORLD peers must be directly accessible in one HCCS domain")
        self.peer_ptrs = torch.tensor(self.handle.buffer_ptrs, dtype=torch.int64, device=self.device)
        self._qkv_staging_key = None
        self._qkv_staging_views = None
        self._closed = False

    def _check_stream(self) -> None:
        if self._closed:
            raise RuntimeError("SHMEM workspace is closed")
        if torch.npu.current_device() != self.device.index:
            raise RuntimeError("SHMEM workspace must be used on its original NPU device")
        if torch.npu.current_stream(self.device) != self.stream:
            raise RuntimeError("SHMEM workspace must be used on a single NPU stream")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("NPU SHMEM Ulysses requires eager execution; graph capture is unsupported")

    def __call__(self, tensor: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
        self._check_stream()
        if tensor.device != self.device:
            raise ValueError("Input must be on the workspace NPU device")
        if tensor.requires_grad:
            raise ValueError("NPU SHMEM Ulysses is inference-only")
        if tensor.ndim != 4 or any(dim <= 0 for dim in tensor.shape):
            raise ValueError("Input must be a nonempty [B, S, H, D] tensor")
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Supported dtypes: bfloat16, float16, float32")
        batch, seq, heads, dim = tensor.shape
        if reverse:
            if seq % self.world:
                raise ValueError("Global sequence length must be divisible by WORLD size")
            seq_local, channels = seq // self.world, heads * dim
            shape = (batch, seq_local, heads * self.world, dim)
        else:
            if heads % self.world:
                raise ValueError("Head count must be divisible by WORLD size; pad/repeat GQA heads upstream")
            seq_local, channels = seq, heads // self.world * dim
            shape = (batch, seq * self.world, heads // self.world, dim)
        required_bytes = tensor.numel() * tensor.element_size()
        if channels * tensor.element_size() % 32:
            raise ValueError("Head shard rows must be 32-byte aligned for the tiled NPU copy")
        if required_bytes > self.capacity_bytes:
            raise ValueError(f"Input requires {required_bytes} bytes; capacity is {self.capacity_bytes}")
        staging = self.allocation[:required_bytes].view(tensor.dtype).view(tensor.shape)
        # copy_ also accepts row-strided Q/K/V views from packed projections.
        staging.copy_(tensor)
        output = torch.empty(shape, dtype=tensor.dtype, device=self.device)
        self._barrier(self.allocation)
        block_channels = triton.next_power_of_2(channels)
        block_rows = max(1, self.block_size // block_channels)
        _pull_and_relayout[(triton.cdiv(seq_local, block_rows), self.world, batch)](
            self.peer_ptrs,
            output,
            self.rank,
            self.world,
            seq_local,
            channels,
            reverse,
            block_rows,
            block_channels,
        )
        self._barrier(self.allocation)
        return output

    def qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        reuse_staging_views: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Exchange Q/K/V together, including unequal Q and KV head counts.

        All three inputs use disjoint symmetric regions. Both stream barriers
        surround the entire group, so no peer can reuse a region prematurely.
        """
        self._check_stream()
        tensors = (query, key, value)
        for tensor in tensors:
            if tensor.device != self.device or tensor.dtype != query.dtype:
                raise ValueError("Q/K/V must share the workspace device and dtype")
            if tensor.requires_grad:
                raise ValueError("NPU SHMEM Ulysses is inference-only")
            if tensor.ndim != 4 or any(dim <= 0 for dim in tensor.shape):
                raise ValueError("Q/K/V must be nonempty [B, S, H, D] tensors")
            if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError("Supported dtypes: bfloat16, float16, float32")
            if tensor.shape[:2] != query.shape[:2] or tensor.shape[2] % self.world:
                raise ValueError("Q/K/V must share batch/sequence dimensions and have divisible head counts")
            if tensor.shape[2] // self.world * tensor.shape[3] * tensor.element_size() % 32:
                raise ValueError("Head shard rows must be 32-byte aligned for the tiled NPU copy")
        sizes = [tensor.numel() for tensor in tensors]
        required_bytes = sum(sizes) * query.element_size()
        if required_bytes > self.capacity_bytes:
            raise ValueError(f"Q/K/V require {required_bytes} bytes; capacity is {self.capacity_bytes}")
        batch, seq = query.shape[:2]
        offsets = (0, sizes[0], sizes[0] + sizes[1])
        staging_key = (query.dtype, tuple(tensor.shape for tensor in tensors))
        if reuse_staging_views and staging_key == self._qkv_staging_key:
            staging_views = self._qkv_staging_views
        else:
            staging = self.allocation[:required_bytes].view(query.dtype)
            staging_views = tuple(
                staging[offset : offset + size].view(tensor.shape)
                for tensor, offset, size in zip(tensors, offsets, sizes)
            )
            if reuse_staging_views:
                # Retain only the last configuration; views alias the fixed
                # allocation, while every call still copies fresh input data.
                self._qkv_staging_key = staging_key
                self._qkv_staging_views = staging_views
        for tensor, staging_view in zip(tensors, staging_views):
            staging_view.copy_(tensor)
        outputs = tuple(
            torch.empty(
                (batch, seq * self.world, tensor.shape[2] // self.world, tensor.shape[3]),
                dtype=tensor.dtype,
                device=self.device,
            )
            for tensor in tensors
        )
        channels = [tensor.shape[2] // self.world * tensor.shape[3] for tensor in tensors]
        block_channels = [triton.next_power_of_2(channel) for channel in channels]
        block_rows = max(1, self.block_size // max(block_channels))
        self._barrier(self.allocation)
        _pull_qkv_and_relayout[(triton.cdiv(seq, block_rows), self.world, batch * 3)](
            self.peer_ptrs,
            *outputs,
            self.rank,
            self.world,
            seq,
            *channels,
            offsets[1],
            offsets[2],
            block_rows,
            *block_channels,
        )
        self._barrier(self.allocation)
        return outputs

    def close(self) -> None:
        """Finish remote reads before collectively releasing the workspace."""
        self._check_stream()
        self._barrier(self.allocation)
        torch.npu.synchronize(self.device)
        self._qkv_staging_views = None
        self._qkv_staging_key = None
        self.peer_ptrs = None
        self.handle = None
        self.allocation = None
        self._closed = True


def get_ulysses_shmem_workspace(group: dist.ProcessGroup, tensors: tuple[torch.Tensor, ...]) -> NPUUlyssesSHMEM:
    """Share one workspace across layers; grow collectively for larger requests.

    Strict Ulysses already requires matching tensor shapes across ranks. The
    resulting byte capacity and growth decisions must consequently match too.
    The group must contain every WORLD rank in the same order, since SHMEM's
    stream barrier is collective over WORLD.
    """
    global _WORKSPACE, _WORKSPACE_GROUP
    if tensors[0].device.type != "npu":
        raise ValueError("NPU ulysses_a2a_permute requires NPU tensors")
    if _WORKSPACE_GROUP is not None and group is not _WORKSPACE_GROUP:
        raise RuntimeError("NPU SHMEM Ulysses supports a single process group; clear it before reinitialization")
    if _WORKSPACE is None:
        if dist.get_process_group_ranks(group) != list(range(dist.get_world_size())):
            raise ValueError("NPU SHMEM Ulysses requires the Ulysses group to match WORLD rank order")
    required = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if _WORKSPACE is not None:
        _WORKSPACE._check_stream()
        if required <= _WORKSPACE.capacity_bytes:
            return _WORKSPACE
        _WORKSPACE.close()
        _WORKSPACE = None
    capacity = ((required + (64 << 20) - 1) // (64 << 20)) * (64 << 20)
    _WORKSPACE = NPUUlyssesSHMEM(capacity)
    _WORKSPACE_GROUP = group
    logger.info(
        "NPU SHMEM Ulysses initialized (rank=%d, world=%d, capacity_bytes=%d)",
        dist.get_rank(),
        dist.get_world_size(),
        capacity,
    )
    return _WORKSPACE


def clear_ulysses_shmem_workspace() -> None:
    """Collectively release the workspace before destroying process groups."""
    global _WORKSPACE, _WORKSPACE_GROUP
    if _WORKSPACE is not None:
        _WORKSPACE.close()
        _WORKSPACE = None
    _WORKSPACE_GROUP = None


def _get_op_workspace(tensors: tuple[torch.Tensor, ...], group_name: str, world_size: int) -> NPUUlyssesSHMEM:
    group = _resolve_process_group(group_name)
    if dist.get_world_size(group) != world_size:
        raise ValueError("world_size must match the Ulysses process group")
    return get_ulysses_shmem_workspace(group, tensors)


def ulysses_qkv_fwd(x: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return _get_op_workspace((x,), group_name, world_size)(x)


def ulysses_qkv_fwd_group(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group_name: str,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _get_op_workspace((query, key, value), group_name, world_size).qkv(query, key, value)


def ulysses_o_rev(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return _get_op_workspace((y,), group_name, world_size)(y, reverse=True)
